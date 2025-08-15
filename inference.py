import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import diffusers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm.auto import tqdm
import math

# Import the model classes from train.py
from train import DiT, TrainingConfig

def load_model_and_config(checkpoint_dir="cifar10_diffusion_ckpt"):
    """Load the trained model and config"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config with weights_only=False for compatibility
    config = torch.load(f"{checkpoint_dir}/config.pth", map_location=device, weights_only=False)
    print(f"📋 Loaded config: {config.hidden_size}d, {config.num_layers}L, {config.num_heads}H")
    
    # Create model
    model = DiT(
        input_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=3,  # RGB for CIFAR-10
        hidden_size=config.hidden_size,
        depth=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=0.0  # No dropout during inference
    )
    
    # Load weights
    state_dict = torch.load(f"{checkpoint_dir}/dit_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, config, device

def generate_images(model, config, device, num_images=16, num_inference_steps=50):
    """Generate images using DDPM sampling"""
    print(f"🎨 Generating {num_images} images with {num_inference_steps} steps...")
    
    # Create noise scheduler for inference
    scheduler = diffusers.DDPMScheduler(num_train_timesteps=200)
    scheduler.set_timesteps(num_inference_steps)
    
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
            
            # Denoising loop
            for t in tqdm(scheduler.timesteps, desc=f"Batch {i//batch_size + 1}"):
                # Prepare timestep
                timestep = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
                
                # Predict noise
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    noise_pred = model(image, timestep)
                
                # Remove noise
                image = scheduler.step(noise_pred, t, image).prev_sample
            
            all_images.append(image.cpu())
    
    # Concatenate all batches
    generated_images = torch.cat(all_images, dim=0)
    
    # Convert from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2
    generated_images = torch.clamp(generated_images, 0, 1)
    
    return generated_images

def save_image_grid(images, filename="generated_images.png", nrow=4):
    """Save images in a grid"""
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    # Convert to PIL Image
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_pil = Image.fromarray((grid_np * 255).astype(np.uint8))
    
    # Save
    grid_pil.save(filename)
    print(f"💾 Saved image grid to {filename}")
    
    return grid_pil

def interpolate_images(model, config, device, num_steps=8):
    """Generate interpolation between two random noise vectors"""
    print(f"🌈 Creating interpolation with {num_steps} steps...")
    
    scheduler = diffusers.DDPMScheduler(num_train_timesteps=200)
    scheduler.set_timesteps(50)
    
    # Create two random noise vectors
    noise1 = torch.randn(1, 3, config.image_size, config.image_size, device=device)
    noise2 = torch.randn(1, 3, config.image_size, config.image_size, device=device)
    
    interpolated_images = []
    
    with torch.no_grad():
        for i in range(num_steps):
            # Linear interpolation
            alpha = i / (num_steps - 1)
            interpolated_noise = (1 - alpha) * noise1 + alpha * noise2
            
            # Denoise
            image = interpolated_noise.clone()
            for t in scheduler.timesteps:
                timestep = torch.full((1,), t, device=device, dtype=torch.long)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    noise_pred = model(image, timestep)
                image = scheduler.step(noise_pred, t, image).prev_sample
            
            # Convert to [0, 1]
            image = (image + 1) / 2
            image = torch.clamp(image, 0, 1)
            interpolated_images.append(image.cpu())
    
    return torch.cat(interpolated_images, dim=0)

def main():
    print("🚀 CIFAR-10 Diffusion Model Inference")
    print("=" * 50)
    
    # Load model
    model, config, device = load_model_and_config()
    
    # Create output directory
    os.makedirs("generated_samples", exist_ok=True)
    
    # Generate regular samples
    print("\n1️⃣ Generating random samples...")
    generated_images = generate_images(model, config, device, num_images=16, num_inference_steps=50)
    save_image_grid(generated_images, "generated_samples/cifar10_samples.png", nrow=4)
    
    # Generate more samples with fewer steps (faster)
    print("\n2️⃣ Generating quick samples (25 steps)...")
    quick_images = generate_images(model, config, device, num_images=16, num_inference_steps=25)
    save_image_grid(quick_images, "generated_samples/cifar10_quick.png", nrow=4)
    
    # Generate interpolation
    print("\n3️⃣ Creating interpolation...")
    interp_images = interpolate_images(model, config, device, num_steps=8)
    save_image_grid(interp_images, "generated_samples/cifar10_interpolation.png", nrow=8)
    
    # Generate high quality samples (more steps)
    print("\n4️⃣ Generating high quality samples (100 steps)...")
    hq_images = generate_images(model, config, device, num_images=8, num_inference_steps=100)
    save_image_grid(hq_images, "generated_samples/cifar10_hq.png", nrow=4)
    
    print("\n🎉 Inference completed!")
    print("📁 Check the 'generated_samples' folder for results:")
    print("   • cifar10_samples.png - Regular samples (50 steps)")
    print("   • cifar10_quick.png - Quick samples (25 steps)")  
    print("   • cifar10_interpolation.png - Noise interpolation")
    print("   • cifar10_hq.png - High quality samples (100 steps)")

if __name__ == "__main__":
    main()