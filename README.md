# Fast CIFAR-10 Diffusion

A high-performance diffusion transformer (DiT) for generating CIFAR-10 images, optimized for RTX 4090.

- YouTube - https://youtu.be/VNuoE0i8lZs

- Bilibili - https://www.bilibili.com/video/BV1zxbWz9Efo/

## Quick Start

```bash
# Train the model
python train.py

# Generate images
python inference.py

# Custom generation
python generate.py --num_images 16 --steps 50
```

## Features

- **DiT Architecture**: Diffusion Transformer with 768d, 12 layers, 12 heads
- **Muon Optimizer**: Hybrid Muon + AdamW optimization
- **High Performance**: Optimized for RTX 4090 with torch.compile, bfloat16, TF32
- **Fast Training**: ~6 minutes for 10 epochs on CIFAR-10

## Requirements

```bash
pip install torch torchvision datasets diffusers accelerate tqdm matplotlib pillow
```

## Model Details

- **Input**: 64x64 RGB images (upscaled from CIFAR-10's 32x32)
- **Patch Size**: 8x8 patches
- **Training**: 10 epochs, batch size 256
- **Inference**: DDPM sampling with 25-100 steps

## Generated Samples

The model generates diverse CIFAR-10 style images including planes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.