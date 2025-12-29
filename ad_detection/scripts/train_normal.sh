#!/bin/bash
# ==============================================================================
# AD Detection Training Script - Normal GPU
# ==============================================================================
# This script trains the AD detection model with the paper's best configuration:
# - Mistral-7B fine-tuned with LoRA
# - VGGish + GRU Autoencoder for acoustic features
# - SVC classifier
#
# Requirements:
# - NVIDIA GPU with at least 16GB VRAM (e.g., RTX 3090, A100, etc.)
# - CUDA installed
#
# Usage:
#   ./scripts/train_normal.sh
#   ./scripts/train_normal.sh --n-folds 3
#   ./scripts/train_normal.sh --train-final
# ==============================================================================

set -e

# Change to script directory
cd "$(dirname "$0")/.."

echo "=============================================="
echo "AD Detection Training - Normal GPU"
echo "=============================================="
echo ""

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# Default parameters
CONFIG="config.yaml"
N_FOLDS=1  # Single train/test split (use --n-folds 5 for full CV)
TRAIN_FINAL=false
OUTPUT_DIR="./output/normal_gpu"

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
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Run training
if [ "$TRAIN_FINAL" = true ]; then
    echo "Training final model on all data..."
    python train.py \
        --config "$CONFIG" \
        --n-folds "$N_FOLDS" \
        --output-dir "$OUTPUT_DIR" \
        --train-final
else
    echo "Running ${N_FOLDS}-fold cross-validation..."
    python train.py \
        --config "$CONFIG" \
        --n-folds "$N_FOLDS" \
        --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "=============================================="
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

