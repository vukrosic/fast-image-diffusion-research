import torch
import diffusers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def load_model_and_scheduler():
    """Load the trained model and create scheduler for inference"""
    # Load the trained UNet model
    model = diffusers.UNet2DModel.from_pretrained("mnist_diffusion_ckpt/unet")
    
    # Create scheduler for inference (same as training but for reverse process)
    scheduler = diffusers.DDPMScheduler(num_train_timesteps=200)
    
    return model, scheduler

def generate_images(model, scheduler, num_images=4, device="cuda"):
    """Generate images using the trained diffusion model"""
    model = model.to(device)
    model.eval()
    
    # Start with random noise
    sample_size = model.config.sample_size
    noise = torch.randn(num_images, 1, sample_size, sample_size).to(device)
    
    # Set scheduler timesteps for inference
    scheduler.set_timesteps(50)  # Use fewer steps for faster inference
    
    # Reverse diffusion process
    with torch.no_grad():
        for t in scheduler.timesteps:
            # Predict noise
            noise_pred = model(noise, t).sample
            
            # Remove predicted noise
            noise = scheduler.step(noise_pred, t, noise).prev_sample
    
    # Convert from [-1, 1] to [0, 1] range
    images = (noise + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    return images

def display_image_in_console(img_array, title="Generated Image"):
    """Display image as ASCII art in console"""
    # Resize to smaller size for console display
    h, w = img_array.shape
    if h > 28 or w > 28:
        # Simple downsampling
        img_array = img_array[::h//28, ::w//28]
    
    # ASCII characters from dark to light
    ascii_chars = " .:-=+*#%@"
    
    print(f"\n{title}:")
    print("-" * (img_array.shape[1] + 2))
    
    for row in img_array:
        line = "|"
        for pixel in row:
            # Map pixel value (0-1) to ASCII character
            char_idx = int(pixel * (len(ascii_chars) - 1))
            line += ascii_chars[char_idx]
        line += "|"
        print(line)
    
    print("-" * (img_array.shape[1] + 2))

def save_and_display_images(images, save_path="generated_mnist.png"):
    """Save and display generated images"""
    num_images = images.shape[0]
    
    # Display each image in console as ASCII
    for i in range(num_images):
        img = images[i].cpu().numpy().squeeze()
        display_image_in_console(img, f"Generated Image {i+1}")
    
    # Also create and save the matplotlib figure
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    if num_images == 1:
        axes = [axes]
    
    for i in range(num_images):
        # Convert tensor to numpy and remove channel dimension
        img = images[i].cpu().numpy().squeeze()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Generated {i+1}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show since we're in console
    print(f"\nImages also saved to {save_path}")

def main():
    # Check if model exists
    if not os.path.exists("mnist_diffusion_ckpt/unet"):
        print("Error: Trained model not found at mnist_diffusion_ckpt/unet")
        print("Make sure you've run the training script first.")
        return
    
    print("Loading trained model...")
    model, scheduler = load_model_and_scheduler()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Generating images...")
    # Generate 8 images
    generated_images = generate_images(model, scheduler, num_images=8, device=device)
    
    print("Saving and displaying results...")
    save_and_display_images(generated_images)
    
    print("Done! Check the generated images.")

if __name__ == "__main__":
    main()