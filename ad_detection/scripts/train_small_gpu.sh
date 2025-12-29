#!/bin/bash
# ==============================================================================
# AD Detection Training Script - Small GPU (4GB VRAM)
# ==============================================================================
# This script trains the AD detection model with optimizations for small GPUs:
# - Smaller model: TinyLlama-1.1B instead of Mistral-7B
# - 4-bit quantization (NF4)
# - Gradient checkpointing
# - Batch size 1 with gradient accumulation
# - Reduced sequence length (512 tokens)
# - Minimal LoRA rank (r=8)
#
# Requirements:
# - NVIDIA GPU with at least 4GB VRAM (e.g., GTX 1650, 1660, RTX 2060)
# - CUDA installed
#
# Tested on:
# - GTX 1650 (4GB) ✓
# - GTX 1660 Ti (6GB) ✓
# - RTX 2060 (6GB) ✓
#
# Usage:
#   ./scripts/train_small_gpu.sh
#   ./scripts/train_small_gpu.sh --n-folds 3
#   ./scripts/train_small_gpu.sh --use-custom-acoustic
# ==============================================================================

set -e

# Change to script directory
cd "$(dirname "$0")/.."

echo "=============================================="
echo "AD Detection Training - Small GPU Mode (4GB+)"
echo "=============================================="
echo "Using TinyLlama-1.1B (optimized for 4GB VRAM)"
echo ""

# Check CUDA and memory
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: True')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'Total memory: {mem:.1f} GB')
    if mem < 4:
        print('WARNING: Less than 4GB VRAM. May run out of memory.')
        print('Consider using --use-custom-acoustic to skip VGGish.')
    elif mem < 6:
        print('4GB mode: Using TinyLlama-1.1B with aggressive optimizations.')
    else:
        print('6GB+ detected: Should run smoothly.')
else:
    print('CUDA not available! Training will be slow on CPU.')
"
echo ""

# Default parameters
CONFIG="config.yaml"
N_FOLDS=1  # Single train/test split (faster)
TRAIN_FINAL=false
USE_CUSTOM_ACOUSTIC=false
OUTPUT_DIR="./output/small_gpu"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --n-folds)
            N_FOLDS="$2"
            shift 2
            ;;
        --train-final)
            TRAIN_FINAL=true
            shift
            ;;
        --use-custom-acoustic)
            USE_CUSTOM_ACOUSTIC=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  N folds: $N_FOLDS"
echo "  Train final: $TRAIN_FINAL"
echo "  Use custom acoustic: $USE_CUSTOM_ACOUSTIC"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Build command
CMD="python train.py --config $CONFIG --n-folds $N_FOLDS --output-dir $OUTPUT_DIR --small-gpu"

if [ "$TRAIN_FINAL" = true ]; then
    CMD="$CMD --train-final"
fi

if [ "$USE_CUSTOM_ACOUSTIC" = true ]; then
    CMD="$CMD --use-custom-acoustic"
fi

echo "Running command: $CMD"
echo ""

# Set environment variables for memory optimization (critical for 4GB)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=0
export TRANSFORMERS_OFFLINE=0

# Clear GPU cache before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

# Run training
eval "$CMD"

echo ""
echo "=============================================="
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

