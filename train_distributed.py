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
    
    # Adjusted for distributed training - increased to better utilize 24GB RTX 4090s
    train_batch_size = 160  # Per GPU batch size (total = 96 * 8 = 768) 
    eval_batch_size = 96   # Per GPU eval batch size
    gradient_accumulation_steps = 1  # Keep as 1 since we have more GPUs
    
    # Increased model size to utilize more GPU memory
    hidden_size = 1024    # Increased from 768 to 1024
    num_layers = 16       # Increased from 12 to 16 layers
    num_heads = 16        # Increased from 12 to 16 heads
    
    # Memory optimization settings
    use_compile = True    # Enable torch.compile for better performance
    enable_amp = True     # Mixed precision training for memory efficiency
    
    # Optimize for RTX 4090s
    num_epochs = 5  # Reduced since we have 8x compute power
    
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
    if dist.is_initialized():
        dist.destroy_process_group()

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
    
    # Compile model for better performance (only on rank 0 to avoid issues)
    if config.compile_model and rank == 0:
        print_rank0("🔥 Compiling model with torch.compile...", rank)
        # Don't compile DDP wrapper, compile the underlying module
        # model.module = torch.compile(model.module, mode='reduce-overhead')
    
    # Training loop
    model.train()
    step = 0
    best_loss = float('inf')
    
    # Progress bar only on rank 0
    pbar = tqdm(total=total_steps, desc="Distributed DiT Training", disable=(rank != 0))
    
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
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
        
        if step >= total_steps:
            break
    
    if rank == 0:
        pbar.close()
    
    training_time = time.time() - start_time
    print_rank0(f"⏱️ Training completed in {training_time/60:.1f} minutes", rank)
    print_rank0(f"🏆 Best loss: {best_loss:.4f}", rank)
    
    # Save model (only from rank 0)
    if rank == 0:
        print_rank0("💾 Saving model...", rank)
        os.makedirs("cifar10_diffusion_distributed_ckpt", exist_ok=True)
        
        # Save the underlying model (not the DDP wrapper)
        torch.save(model.module.state_dict(), "cifar10_diffusion_distributed_ckpt/dit_model.pth")
        torch.save(config, "cifar10_diffusion_distributed_ckpt/config.pth")
        
        print_rank0("💾 Model saved to cifar10_diffusion_distributed_ckpt/", rank)
        
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
                f"cifar10_diffusion_distributed_ckpt/sample_{CIFAR10_CLASSES[class_idx]}.png",
                normalize=True,
                value_range=(-1, 1),
                nrow=2
            )
    
    # Cleanup
    cleanup_distributed()
    
    print_rank0("🎉 DISTRIBUTED TRAINING COMPLETED!", rank)
    return model

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
        cleanup_distributed()

if __name__ == "__main__":
    main()
