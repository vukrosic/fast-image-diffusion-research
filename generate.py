#!/usr/bin/env python3
"""
Simple script to generate images with the trained CIFAR-10 diffusion model
Usage: python generate.py --num_images 16 --steps 50
"""

import argparse
import torch
import torchvision
from PIL import Image
import os
from inference import load_model_and_config, generate_images, save_image_grid

def main():
    parser = argparse.ArgumentParser(description='Generate CIFAR-10 images with trained diffusion model')
    parser.add_argument('--num_images', type=int, default=16, help='Number of images to generate')
    parser.add_argument('--steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--output', type=str, default='generated.png', help='Output filename')
    parser.add_argument('--checkpoint', type=str, default='cifar10_diffusion_ckpt', help='Checkpoint directory')
    parser.add_argument('--nrow', type=int, default=4, help='Number of images per row in grid')
    
    args = parser.parse_args()
    
    print(f"🎨 Generating {args.num_images} images with {args.steps} steps...")
    
    # Load model
    model, config, device = load_model_and_config(args.checkpoint)
    
    # Generate images
    images = generate_images(model, config, device, args.num_images, args.steps)
    
    # Save grid
    save_image_grid(images, args.output, nrow=args.nrow)
    
    print(f"✅ Generated {args.num_images} images saved to {args.output}")

if __name__ == "__main__":
    main()