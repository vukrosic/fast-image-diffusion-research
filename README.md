# Fast Image Diffusion Research

- YouTube - https://youtu.be/ZjTBcC8PYMo

- Bilibili - https://www.bilibili.com/video/BV1szevzVEEm/

I recommend using this in AI code IDE and asking it questions about it.

Gemini CLI or Client + Cerebras is free.

A fast implementation of diffusion models for CIFAR-10 image generation using DiT (Diffusion Transformer) architecture.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model

**Single GPU Training:**
```bash
python train.py
```

**Distributed Training (8x RTX 4090):**
```bash
# Option 1: Using torchrun (recommended)
torchrun --nproc_per_node=8 train_distributed.py

# Option 2: Using launch script
bash launch_distributed.sh
```

### 3. Generate Images

**From Text Prompts:**
```bash
python text_to_image.py --prompt "bird" --num_images 4 --steps 50
```

**Available Prompts:**
- `airplane`, `plane`, `jet`
- `car`, `automobile`, `vehicle`
- `bird`, `eagle`, `duck`
- `cat`, `feline`, `lion`
- `deer`, `stag`, `moose`
- `dog`, `canine`, `puppy`
- `frog`, `toad`
- `horse`, `mare`, `pony`
- `ship`, `boat`, `yacht`
- `truck`, `lorry`, `van`

## Model Architecture

- **Architecture**: DiT (Diffusion Transformer)
- **Dataset**: CIFAR-10 (32x32 images)
- **Model Size**: 1024 hidden dim, 16 layers, 16 heads
- **Training**: 200 epochs with mixed precision

## Checkpoints

- **Single GPU**: `cifar10_diffusion_ckpt/`
- **Distributed**: `cifar10_diffusion_distributed_ckpt/`

## GPU Requirements

- **Single GPU**: 24GB+ VRAM (RTX 3090/4090)
- **Distributed**: 8x RTX 4090 (24GB each)

## Monitoring

Monitor GPU usage during training:
```bash
python gpu_monitor.py
```
