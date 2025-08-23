#!/usr/bin/env python3
"""
Text-to-image generation script for CIFAR-10 diffusion model
Maps common words to CIFAR-10 categories and generates corresponding images

Usage: 
    python text_to_image.py --prompt "bird" --num_images 4 --steps 50
    python text_to_image.py --prompt "car" --num_images 8 --steps 25
"""

import argparse
import torch
import torchvision
from PIL import Image
import os
from train import TrainingConfig, CIFAR10_CLASSES, DiT, get_noise_scheduler
import diffusers
import sys
import importlib.util

# Word mappings to CIFAR-10 classes
WORD_MAPPINGS = {
    # airplane
    'airplane': 0, 'plane': 0, 'jet': 0, 'aircraft': 0, 'flying machine': 0,
    
    # automobile
    'automobile': 1, 'car': 1, 'vehicle': 1, 'auto': 1, 'sedan': 1, 
    'hatchback': 1, 'convertible': 1, 'wagon': 1, 'motorcar': 1,
    
    # bird
    'bird': 2, 'avian': 2, 'fowl': 2, 'sparrow': 2, 'eagle': 2,
    'robin': 2, 'duck': 2, 'goose': 2, 'pigeon': 2, 'owl': 2,
    
    # cat
    'cat': 3, 'feline': 3, 'kitten': 3, 'tabby': 3, 'persian cat': 3,
    'siamese cat': 3, 'lion': 3, 'tiger': 3, 'leopard': 3, 'cheetah': 3,
    
    # deer
    'deer': 4, 'stag': 4, 'buck': 4, 'doe': 4, 'reindeer': 4, 
    'elk': 4, 'moose': 4, 'antelope': 4,
    
    # dog
    'dog': 5, 'canine': 5, 'puppy': 5, 'hound': 5, 'poodle': 5,
    'beagle': 5, 'shepherd': 5, 'retriever': 5, 'terrier': 5, 'bulldog': 5,
    
    # frog
    'frog': 6, 'toad': 6, 'amphibian': 6, 'bullfrog': 6, 'tadpole': 6,
    
    # horse
    'horse': 7, 'mare': 7, 'stallion': 7, 'pony': 7, 'colt': 7,
    'equine': 7, 'mustang': 7, 'thoroughbred': 7,
    
    # ship
    'ship': 8, 'boat': 8, 'vessel': 8, 'sailboat': 8, 'cruiser': 8,
    'yacht': 8, 'barge': 8, 'ferry': 8, 'submarine': 8, 'warship': 8,
    
    # truck
    'truck': 9, 'lorry': 9, 'pickup': 9, 'van': 9, 'semi': 9,
    'tractor': 9, 'fire truck': 9, 'dump truck': 9, 'tanker': 9,
}

def map_word_to_class(word):
    """Map a word to a CIFAR-10 class index"""
    word = word.lower().strip()
    if word in WORD_MAPPINGS:
        return WORD_MAPPINGS[word]
    else:
        # Try to find partial matches
        for key in WORD_MAPPINGS:
            if word in key or key in word:
                return WORD_MAPPINGS[key]
    
    # Return None if no match found
    return None

def load_model_and_config(checkpoint_dir):
    """Load trained model and config from checkpoint directory"""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # First try to load from distributed training checkpoint (latest_checkpoint.pth)
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_checkpoint_path):
        print(f"📥 Loading distributed training checkpoint from {latest_checkpoint_path}")
        
        # Load checkpoint with custom class mapping to avoid DistributedTrainingConfig issues
        try:
            # Temporarily add DistributedTrainingConfig to globals to avoid loading errors
            import train_distributed
            sys.modules['__main__'].DistributedTrainingConfig = train_distributed.DistributedTrainingConfig
            
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            
            # Clean up the temporary class
            if 'DistributedTrainingConfig' in sys.modules['__main__'].__dict__:
                del sys.modules['__main__'].DistributedTrainingConfig
                
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint with class mapping: {e}")
            print("🔄 Trying alternative loading method...")
            
            # Try loading just the model weights directly
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False, pickle_module=torch._utils._rebuild_tensor_v2)
        
        # Extract config from checkpoint
        config = checkpoint.get('config')
        if config is None:
            # Try loading config separately if not in checkpoint
            config_path = os.path.join(checkpoint_dir, 'config.pth')
            if os.path.exists(config_path):
                config = torch.load(config_path, map_location=device, weights_only=False)
            else:
                raise FileNotFoundError(f"Config not found in checkpoint or separate file")
        
        # Handle distributed training config if it's a DistributedTrainingConfig
        if hasattr(config, '__class__') and config.__class__.__name__ == 'DistributedTrainingConfig':
            print("🔄 Converting DistributedTrainingConfig to TrainingConfig for inference...")
            # Create a new TrainingConfig with the same parameters
            from train import TrainingConfig
            base_config = TrainingConfig()
            # Copy relevant attributes
            for attr in ['image_size', 'patch_size', 'hidden_size', 'num_layers', 
                        'num_heads', 'mlp_ratio', 'dropout', 'num_classes', 'cfg_scale']:
                if hasattr(config, attr):
                    setattr(base_config, attr, getattr(config, attr))
            config = base_config
        
        # If config is still problematic, try to create a default config
        if not hasattr(config, 'image_size') or not hasattr(config, 'hidden_size'):
            print("⚠️  Config appears to be corrupted, using default values...")
            from train import TrainingConfig
            config = TrainingConfig()
        
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
        )
        
        # Load model weights from checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        return model, config, device
    
    # Fallback to old format (separate files)
    else:
        print(f"📥 Loading from separate checkpoint files in {checkpoint_dir}")
        
        # Load config
        config_path = os.path.join(checkpoint_dir, 'config.pth')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = torch.load(config_path, map_location=device, weights_only=False)
        
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
        )
        
        # Load model weights
        model_path = os.path.join(checkpoint_dir, 'dit_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model = model.to(device)
        model.eval()
        
        return model, config, device

def save_image_grid(images, filename, nrow=4):
    """Save a grid of images"""
    torchvision.utils.save_image(images, filename, nrow=nrow, normalize=False, value_range=(0, 1))

def generate_images(model, scheduler, class_idx, num_images, cfg_scale, device):
    """Generate images for a specific class"""
    with torch.no_grad():
        # Start with random noise
        shape = (num_images, 3, model.input_size, model.input_size)
        image = torch.randn(shape, device=device)
        
        # Create label tensor
        labels = torch.full((num_images,), class_idx, device=device, dtype=torch.long)
        
        # Denoising loop
        for t in scheduler.timesteps:
            timestep = torch.full((num_images,), t, device=device, dtype=torch.long)
            
            # Predict noise with and without conditioning
            noise_pred_cond = model(image, timestep, labels)
            noise_pred_uncond = model(image, timestep, None)
            
            # Apply classifier-free guidance
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Remove noise
            image = scheduler.step(noise_pred, t, image).prev_sample
    
    return image

def generate_images_from_text(prompt, num_images=4, steps=50, checkpoint_dir="cifar10_diffusion_distributed_ckpt"):
    """Generate images based on text prompt"""
    print(f"🎨 Generating images for prompt: '{prompt}'")
    
    # Map word to class
    class_idx = map_word_to_class(prompt)
    if class_idx is None:
        print(f"❌ No mapping found for '{prompt}'")
        print("Available classes:")
        for i, class_name in enumerate(CIFAR10_CLASSES):
            print(f"  {i}: {class_name}")
        return None
    
    class_name = CIFAR10_CLASSES[class_idx]
    print(f"✅ Mapped '{prompt}' to CIFAR-10 class '{class_name}' (index {class_idx})")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory '{checkpoint_dir}' not found!")
        print("Please train the model first using: python train.py")
        return None
    
    # Load model
    try:
        model, config, device = load_model_and_config(checkpoint_dir)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Please train the model first using: python train.py")
        return None
    
    # Create noise scheduler for inference
    scheduler = diffusers.DDPMScheduler(num_train_timesteps=200)
    scheduler.set_timesteps(steps)
    
    # Generate random noise
    batch_size = min(num_images, 8)  # Process in batches to save memory
    all_images = []
    
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            current_batch_size = min(batch_size, num_images - i)
            
            # Start with random noise
            image = torch.randn(
                current_batch_size, 3, config.image_size, config.image_size,
                device=device, dtype=torch.float32
            )
            
            # Create label tensor
            labels = torch.full((current_batch_size,), class_idx, device=device, dtype=torch.long)
            
            # Denoising loop with classifier-free guidance
            for t in scheduler.timesteps:
                # Prepare timestep
                timestep = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
                
                # Predict noise with and without conditioning
                noise_pred_cond = model(image, timestep, labels)
                noise_pred_uncond = model(image, timestep, None)
                
                # Apply classifier-free guidance
                noise_pred = noise_pred_uncond + config.cfg_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Remove noise
                image = scheduler.step(noise_pred, t, image).prev_sample
            
            all_images.append(image.cpu())
    
    # Concatenate all batches
    generated_images = torch.cat(all_images, dim=0)
    
    # Convert from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2
    generated_images = torch.clamp(generated_images, 0, 1)
    
    return generated_images

def main():
    parser = argparse.ArgumentParser(description='Generate CIFAR-10 images from text prompts')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt (e.g., "bird", "car", "airplane")')
    parser.add_argument('--num_images', type=int, default=4, help='Number of images to generate')
    parser.add_argument('--steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--output', type=str, default='text_generated.png', help='Output filename')
    parser.add_argument('--checkpoint', type=str, default='cifar10_diffusion_distributed_ckpt', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Generate images
    images = generate_images_from_text(
        args.prompt, 
        args.num_images, 
        args.steps, 
        args.checkpoint
    )
    
    if images is not None:
        # Save grid
        nrow = min(args.num_images, 4)  # At most 4 images per row
        save_image_grid(images, args.output, nrow=nrow)
        print(f"✅ Generated images saved to {args.output}")
    else:
        print("❌ Image generation failed!")

if __name__ == "__main__":
    main()
