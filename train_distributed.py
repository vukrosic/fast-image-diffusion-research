#!/usr/bin/env python3
"""
Distributed training script for CIFAR-10 diffusion model on 8x RTX 4090 GPUs
Uses PyTorch DistributedDataParallel (DDP) for efficient multi-GPU training

Usage:
    # Launch with torchrun (recommended)
    torchrun --nproc_per_node=8 train_distributed.py
    
    # Or launch with python -m torch.distributed.launch (deprecated but still works)
    python -m torch.distributed.launch --nproc_per_node=8 train_distributed.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
import datasets
import diffusers
import accelerate
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import os
import math
import random
import numpy as np
from dataclasses import dataclass
import warnings
import time

warnings.filterwarnings('ignore')

# Import from the original training script
from train import (
    set_seed, zeropower_via_newtonschulz5, Muon, TrainingConfig, 
    CIFAR10_CLASSES, transform, PatchEmbed, TimestepEmbedder, 
    DiTBlock, DiT, setup_muon_optimizer, get_noise_scheduler,
    get_lr_schedulers, sample_with_label
)

@dataclass
class DistributedTrainingConfig(TrainingConfig):
    """Extended config for distributed training"""
    # Distributed training parameters
    world_size: int = 8  # Number of GPUs
    local_rank: int = 0  # Will be set by launcher
    
        # Adjusted for distributed training - 50% larger batch for better convergence
    train_batch_size = 192  # Per GPU batch size (total = 192 * 8 = 1536) - 50% larger
    eval_batch_size = 96    # Per GPU eval batch size
    gradient_accumulation_steps = 1  # Keep as 1 since we have more GPUs
    
    # Increased model size to utilize more GPU memory
    hidden_size = 1024    # Increased from 768 to 1024
    num_layers = 16       # Increased from 12 to 16 layers
    num_heads = 16        # Increased from 12 to 16 heads
    
    # Memory optimization settings
    use_compile = True    # Enable torch.compile for better performance
    enable_amp = True     # Mixed precision training for memory efficiency
    use_gradient_checkpointing = True  # Trade compute for memory
    
    # Extended training for better convergence
    num_epochs = 200  # Increased 10x for longer training (was 5)
    
    # Resume training settings
    resume_from_checkpoint = ""  # Path to checkpoint directory to resume from
    checkpoint_dir = "cifar10_diffusion_distributed_ckpt"  # Where to save checkpoints
    
    # Debug settings
    save_debug_samples = True  # Save sample images with labels for verification
    debug_samples_count = 16   # Number of debug samples to save
    
    def __post_init__(self):
        super().__post_init__()
        # Set local rank from environment if available
        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Environment variables for distributed training not found!")
        print("Please use torchrun or torch.distributed.launch")
        return False, 0, 1, 0
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU communication
        init_method='env://',  # Use environment variables
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    return True, rank, world_size, local_rank

def cleanup_distributed():
    """Clean up distributed training"""
    try:
        # Clear GPU cache before cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        # Ignore cleanup errors - they're common and usually harmless
        pass

def save_checkpoint(model, optimizers, schedulers, epoch, config, checkpoint_dir, rank):
    """Save complete checkpoint including model, optimizer, and scheduler states"""
    if rank == 0:  # Only save from rank 0
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': [opt.state_dict() for opt in optimizers],
            'scheduler_state_dict': [sch.state_dict() for sch in schedulers] if schedulers else None,
            'epoch': epoch,
            'config': config,
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save model and config separately for compatibility with text_to_image.py
        torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, "dit_model.pth"))
        torch.save(config, os.path.join(checkpoint_dir, "config.pth"))
        
        print(f"💾 Checkpoint saved at epoch {epoch} to {checkpoint_dir}")

def load_checkpoint(checkpoint_path, model, optimizers, schedulers, device, rank):
    """Load checkpoint and restore model, optimizer, and scheduler states"""
    if not os.path.exists(checkpoint_path):
        print_rank0(f"❌ Checkpoint not found: {checkpoint_path}", rank)
        return 0
    
    print_rank0(f"📥 Loading checkpoint from {checkpoint_path}", rank)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.module.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizers and 'optimizer_state_dict' in checkpoint:
        optimizer_states = checkpoint['optimizer_state_dict']
        for i, opt in enumerate(optimizers):
            if i < len(optimizer_states):
                opt.load_state_dict(optimizer_states[i])
    
    # Load scheduler state
    if schedulers and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler_states = checkpoint['scheduler_state_dict']
        for i, sch in enumerate(schedulers):
            if i < len(scheduler_states):
                sch.load_state_dict(scheduler_states[i])
    
    epoch = checkpoint.get('epoch', 0)
    print_rank0(f"✅ Resumed from epoch {epoch}", rank)
    
    return epoch + 1  # Return next epoch to continue from

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_path):
        return latest_path
    
    # Fallback: look for numbered checkpoints
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if checkpoint_files:
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join(checkpoint_dir, checkpoint_files[-1])
    
    return None

def save_debug_samples(images, labels, epoch, step, checkpoint_dir, rank):
    """Save sample images with their labels for debugging data pipeline"""
    if rank != 0:  # Only save from rank 0
        return
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create debug directory
    debug_dir = os.path.join(checkpoint_dir, "debug_samples")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Convert tensors to numpy and denormalize
    images_np = images.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    # Denormalize images from [-1, 1] to [0, 1]
    images_np = (images_np + 1.0) / 2.0
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # NCHW -> NHWC
    
    # Create a grid of images with labels
    num_samples = min(len(images_np), 16)  # Show up to 16 samples
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'Training Data Debug - Epoch {epoch+1}, Step {step}', fontsize=16)
    
    for i in range(16):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        
        if i < num_samples:
            # Show image
            ax.imshow(images_np[i])
            
            # Add label information
            label_idx = labels_np[i]
            class_name = CIFAR10_CLASSES[label_idx]
            ax.set_title(f'Label: {label_idx}\nClass: {class_name}', fontsize=10)
            
            # Add colored border based on class
            colors = ['red', 'blue', 'green', 'orange', 'purple', 
                     'brown', 'pink', 'gray', 'olive', 'cyan']
            border_color = colors[label_idx % len(colors)]
            rect = patches.Rectangle((0, 0), 31, 31, linewidth=2, 
                                   edgecolor=border_color, facecolor='none')
            ax.add_patch(rect)
        else:
            ax.axis('off')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save the debug image
    debug_filename = f"debug_epoch_{epoch+1:03d}_step_{step:05d}.png"
    debug_path = os.path.join(debug_dir, debug_filename)
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save a text file with detailed label information
    txt_filename = f"debug_epoch_{epoch+1:03d}_step_{step:05d}.txt"
    txt_path = os.path.join(debug_dir, txt_filename)
    
    with open(txt_path, 'w') as f:
        f.write(f"Training Data Debug - Epoch {epoch+1}, Step {step}\n")
        f.write("=" * 50 + "\n\n")
        
        for i in range(num_samples):
            label_idx = labels_np[i]
            class_name = CIFAR10_CLASSES[label_idx]
            f.write(f"Image {i+1:2d}: Label={label_idx}, Class='{class_name}'\n")
        
        f.write(f"\nCIFAR-10 Classes Reference:\n")
        for idx, class_name in enumerate(CIFAR10_CLASSES):
            f.write(f"  {idx}: {class_name}\n")
    
    print(f"🔍 Debug samples saved: {debug_path}")
    print(f"📝 Debug info saved: {txt_path}")

def get_distributed_dataloader(config: DistributedTrainingConfig, world_size: int, rank: int):
    """Create distributed dataloader for CIFAR-10"""
    # Load CIFAR-10 dataset
    cifar10_dataset = datasets.load_dataset('cifar10', split='train')
    cifar10_dataset.reset_format()
    cifar10_dataset.set_transform(transform)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        cifar10_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True  # Important for consistent batch sizes across GPUs
    )
    
    # Create dataloader with distributed sampler
    dataloader = torch.utils.data.DataLoader(
        cifar10_dataset,
        batch_size=config.train_batch_size,  # Per-GPU batch size
        sampler=sampler,
        num_workers=4,  # Reduced per GPU since we have 8 GPUs
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,  # Reduced prefetch factor
        drop_last=True
    )
    
    return dataloader, cifar10_dataset, sampler

def print_rank0(message, rank=0):
    """Print message only from rank 0 to avoid spam"""
    if rank == 0:
        print(message)

def train_distributed(config: DistributedTrainingConfig):
    """Main distributed training function"""
    
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    if not is_distributed:
        return None
    
    config.local_rank = local_rank
    config.world_size = world_size
    
    # Set device
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    print_rank0(f"\n🚀 Starting Distributed Training on {world_size} RTX 4090 GPUs", rank)
    print_rank0(f"📊 Global batch size: {config.train_batch_size * world_size}", rank)
    print_rank0(f"📊 Per-GPU batch size: {config.train_batch_size}", rank)
    
    # Set seeds (different for each rank for data diversity)
    set_seed(config.seed + rank)
    
    # Load distributed data
    train_dataloader, cifar10_dataset, sampler = get_distributed_dataloader(config, world_size, rank)
    print_rank0(f"📊 Dataset: {len(cifar10_dataset)} samples, {len(train_dataloader)} batches per GPU", rank)
    
    # Create model
    model = DiT(
        input_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=3,
        hidden_size=config.hidden_size,
        depth=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        num_classes=config.num_classes
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    # Print model info from rank 0
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total parameters: {total_params:,}")
        
        # Test model shapes
        sample_image = cifar10_dataset[0]["images"].unsqueeze(0).to(device)
        sample_label = torch.tensor([cifar10_dataset[0]["labels"]]).to(device)
        with torch.no_grad():
            test_output = model(sample_image, torch.tensor([0]).to(device), sample_label)
            print(f"✅ Model test - Input: {sample_image.shape}, Output: {test_output.shape}")
    
    # Setup noise scheduler
    noise_scheduler = get_noise_scheduler()
    
    # Calculate training steps
    steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    print_rank0(f"📊 Training: {config.num_epochs} epochs, {total_steps:,} steps", rank)
    
    # Setup optimizers and schedulers
    optimizers = setup_muon_optimizer(model.module, config)  # Use .module for DDP
    schedulers = get_lr_schedulers(optimizers, config, total_steps)
    
    # Mixed precision setup - use bfloat16 for RTX 4090
    scaler = GradScaler() if config.use_amp and config.mixed_precision == 'fp16' else None
    use_autocast = config.use_amp
    autocast_dtype = torch.bfloat16 if config.mixed_precision == 'bf16' else torch.float16
    
    print_rank0(f"🔥 Using mixed precision: {config.mixed_precision}, autocast: {use_autocast}", rank)
    
    # Check for resume checkpoint
    start_epoch = 0
    if config.resume_from_checkpoint:
        checkpoint_path = config.resume_from_checkpoint
        if os.path.isdir(checkpoint_path):
            checkpoint_path = find_latest_checkpoint(checkpoint_path)
        if checkpoint_path:
            start_epoch = load_checkpoint(checkpoint_path, model, optimizers, schedulers, device, rank)
        else:
            print_rank0(f"⚠️  No checkpoint found at {config.resume_from_checkpoint}, starting from scratch", rank)
    elif os.path.exists(config.checkpoint_dir):
        # Auto-resume from latest checkpoint if available
        checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        if checkpoint_path:
            print_rank0("🔄 Found existing checkpoint, resuming automatically...", rank)
            start_epoch = load_checkpoint(checkpoint_path, model, optimizers, schedulers, device, rank)
    
    # Clear cache after checkpoint loading
    torch.cuda.empty_cache()
    print_rank0(f"🧹 GPU memory cleared after checkpoint loading", rank)
    
    # Compile model for better performance (only on rank 0 to avoid issues)
    if config.compile_model and rank == 0:
        print_rank0("🔥 Compiling model with torch.compile...", rank)
        # Don't compile DDP wrapper, compile the underlying module
        # model.module = torch.compile(model.module, mode='reduce-overhead')
    
    # Training loop
    model.train()
    step = 0
    best_loss = float('inf')
    debug_samples_saved = False  # Track if we've saved debug samples this session
    
    # Progress bar only on rank 0
    pbar = tqdm(total=total_steps, desc="Distributed DiT Training", disable=(rank != 0))
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config.num_epochs):
        print_rank0(f"\n📅 Epoch {epoch + 1}/{config.num_epochs}", rank)
        
        # Set epoch for DistributedSampler
        sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        for batch in train_dataloader:
            clean_images = batch['images'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            noise = torch.randn_like(clean_images)
            batch_size = clean_images.shape[0]
            
            # Save debug samples for data verification (first few batches of this session)
            if config.save_debug_samples and not debug_samples_saved and epoch_steps < 3:  # Save first 3 batches of current session
                save_debug_samples(clean_images[:config.debug_samples_count], 
                                 labels[:config.debug_samples_count], 
                                 epoch, epoch_steps, config.checkpoint_dir, rank)
                if epoch_steps >= 2:  # Mark as saved after 3 batches (0, 1, 2)
                    debug_samples_saved = True
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, 
                (batch_size,), device=device
            )
            
            # Add noise to images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps.cpu())
            noisy_images = noisy_images.to(device, non_blocking=True)
            
            # Randomly drop labels for classifier-free guidance
            if random.random() < config.cfg_prob:
                labels = None
            
            # Forward pass
            if use_autocast:
                with autocast(dtype=autocast_dtype):
                    noise_pred = model(noisy_images, timesteps, labels)
                    loss = F.mse_loss(noise_pred, noise)
                    loss = loss / config.gradient_accumulation_steps
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                noise_pred = model(noisy_images, timesteps, labels)
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
            
            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    # Unscale gradients for clipping
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    
                    # Step optimizers
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    
                    # Step schedulers
                    for scheduler in schedulers:
                        scheduler.step()
                    
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
            
            # Accumulate loss for logging
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            epoch_steps += 1
            
            # Logging (only from rank 0)
            if step % 100 == 0 and rank == 0:
                current_loss = loss.item() * config.gradient_accumulation_steps
                lr = optimizers[0].param_groups[0]["lr"]
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': step,
                    'epoch': epoch + 1
                })
                
                if current_loss < best_loss:
                    best_loss = current_loss
            
            step += 1
            if step % 100 == 0 and rank == 0:
                pbar.update(100)
            
            if step >= total_steps:
                break
        
        # Log epoch statistics
        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss / epoch_steps
            print_rank0(f"📊 Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}", rank)
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_checkpoint(model, optimizers, schedulers, epoch, config, config.checkpoint_dir, rank)
        
        if step >= total_steps:
            break
    
    if rank == 0:
        pbar.close()
    
    training_time = time.time() - start_time
    print_rank0(f"⏱️ Training completed in {training_time/60:.1f} minutes", rank)
    print_rank0(f"🏆 Best loss: {best_loss:.4f}", rank)
    
    # Final save model (only from rank 0)
    if rank == 0:
        print_rank0("💾 Saving final model...", rank)
        save_checkpoint(model, optimizers, schedulers, config.num_epochs - 1, config, config.checkpoint_dir, rank)
        
        # Demo: Generate samples for a few classes
        print_rank0("\n🎨 Generating sample images...", rank)
        model.eval()
        
        for class_idx in [0, 2, 5]:  # airplane, bird, dog
            print_rank0(f"Generating {CIFAR10_CLASSES[class_idx]} (class {class_idx})...", rank)
            samples = sample_with_label(
                model.module,  # Use underlying model, not DDP wrapper
                noise_scheduler,
                label=class_idx,
                num_samples=2,
                cfg_scale=config.cfg_scale,
                device=device
            )
            
            # Save sample
            torchvision.utils.save_image(
                samples,
                f"{config.checkpoint_dir}/sample_{CIFAR10_CLASSES[class_idx]}.png",
                normalize=True,
                value_range=(-1, 1),
                nrow=2
            )
    
    # Clear model and cleanup memory before returning
    del model, optimizers, schedulers, train_dataloader
    torch.cuda.empty_cache()
    
    print_rank0("🎉 DISTRIBUTED TRAINING COMPLETED!", rank)
    return None  # Don't return model to save memory

def main():
    """Main function"""
    print("🚀 CIFAR-10 Diffusion Model - Distributed Training on 8x RTX 4090")
    
    # Check if we're in a distributed environment
    if 'RANK' not in os.environ:
        print("❌ This script requires distributed training setup!")
        print("Please use one of these commands:")
        print("  torchrun --nproc_per_node=8 train_distributed.py")
        print("  python -m torch.distributed.launch --nproc_per_node=8 train_distributed.py")
        return
    
    # Get rank info
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print_rank0(f"🔍 Detected {world_size} GPUs for distributed training", rank)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print_rank0("❌ CUDA not available!", rank)
        return
    
    if rank == 0:
        print(f"🔍 Available GPUs: {torch.cuda.device_count()}")
        for i in range(min(torch.cuda.device_count(), 8)):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
    
    # Enable optimizations for RTX 4090
    if rank == 0:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print_rank0("🚀 Enabled TF32 for faster training", rank)
    
    # Create config
    config = DistributedTrainingConfig()
    
    if rank == 0:
        print(f"\n📋 Distributed Training Configuration:")
        print(f"   GPUs: {config.world_size}")
        print(f"   Global batch size: {config.train_batch_size * config.world_size}")
        print(f"   Per-GPU batch size: {config.train_batch_size}")
        print(f"   Architecture: {config.hidden_size}d, {config.num_layers}L, {config.num_heads}H")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Classes: {config.num_classes}, CFG prob: {config.cfg_prob}")
    
    # Start training
    try:
        model = train_distributed(config)
        if rank == 0:
            print_rank0("✅ Training completed successfully!", rank)
    except Exception as e:
        print_rank0(f"❌ Training failed: {e}", rank)
        raise
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        cleanup_distributed()

if __name__ == "__main__":
    main()
