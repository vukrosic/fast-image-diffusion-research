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
from inference import load_model_and_config, generate_images, save_image_grid
from train import TrainingConfig, CIFAR10_CLASSES

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

def generate_images_from_text(prompt, num_images=4, steps=50, checkpoint_dir="cifar10_diffusion_ckpt"):
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
    parser.add_argument('--checkpoint', type=str, default='cifar10_diffusion_ckpt', help='Checkpoint directory')
    
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
