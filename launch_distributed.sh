#!/bin/bash
# Launch script for distributed training on 8x RTX 4090 GPUs

echo "🚀 Launching Distributed Training on 8x RTX 4090 GPUs"
echo "📊 Total batch size: 1536 (192 per GPU)"
echo "⚡ Using PyTorch DistributedDataParallel (DDP)"

# Check if GPUs are available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

echo "🔍 Available GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

# Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "📊 Detected $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -lt 8 ]; then
    echo "⚠️  Warning: Only $GPU_COUNT GPUs detected, but script is configured for 8 GPUs"
    echo "You can modify the --nproc_per_node parameter in this script"
fi

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0

# Launch distributed training with torchrun (recommended for PyTorch 1.10+)
echo "🚀 Starting distributed training..."
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_distributed.py

echo "✅ Training launch completed!"
