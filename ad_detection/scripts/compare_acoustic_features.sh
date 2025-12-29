#!/bin/bash
# ==============================================================================
# Compare Acoustic Feature Methods for AD Detection
# ==============================================================================
# This script compares three approaches:
#
# 1. LLM-only: Uses only Mistral-7B text features (no acoustic)
# 2. VGGish + GRU-AE: Paper method with VGGish embeddings encoded by GRU autoencoder
# 3. Custom Acoustic: Uses features from audio_feature_extraction pipeline
#    (VAD timing, prosody, voice quality - 42 clinician-interpretable features)
#
# The comparison helps understand:
# - How much acoustic features improve over text-only
# - Whether custom interpretable features perform as well as deep embeddings
#
# Usage:
#   ./scripts/compare_acoustic_features.sh              # Uses TinyLlama, 1 fold
#   ./scripts/compare_acoustic_features.sh --n-folds 5  # Full 5-fold CV
#   ./scripts/compare_acoustic_features.sh --no-small-gpu  # Use Mistral-7B (needs 16GB+)
# ==============================================================================

set -e

# Change to script directory
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Acoustic Feature Comparison"
echo "=============================================="
echo ""
echo "Comparing:"
echo "  1. LLM-only (Mistral-7B)"
echo "  2. VGGish + GRU-AE (paper method)"
echo "  3. Custom Acoustic Features (interpretable)"
echo ""

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Default parameters
CONFIG="config.yaml"
N_FOLDS=1  # Single train/test split (faster)
SMALL_GPU=true  # Use TinyLlama for 4GB GPUs
OUTPUT="comparison_results.json"

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
        --small-gpu)
            SMALL_GPU=true
            shift
            ;;
        --no-small-gpu)
            SMALL_GPU=false
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  N folds: $N_FOLDS"
echo "  Small GPU mode: $SMALL_GPU"
echo "  Output file: $OUTPUT"
echo ""

# Check for custom features file
CUSTOM_FEATURES="../audio_feature_extraction/output/pitt_audio_features.csv"
if [ -f "$CUSTOM_FEATURES" ]; then
    echo "Found custom acoustic features: $CUSTOM_FEATURES"
    FEATURE_COUNT=$(head -1 "$CUSTOM_FEATURES" | tr ',' '\n' | wc -l)
    SAMPLE_COUNT=$(wc -l < "$CUSTOM_FEATURES")
    echo "  Samples: $((SAMPLE_COUNT - 1))"
    echo "  Features: $FEATURE_COUNT"
else
    echo "WARNING: Custom acoustic features not found at $CUSTOM_FEATURES"
    echo "Run audio_feature_extraction first to generate features."
    echo "The comparison will only include LLM-only and VGGish+GRU methods."
fi
echo ""

# Build command
CMD="python compare_features.py --config $CONFIG --n-folds $N_FOLDS --output $OUTPUT"

if [ "$SMALL_GPU" = true ]; then
    CMD="$CMD --small-gpu"
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
    echo "Model: TinyLlama-1.1B (small GPU mode)"
else
    echo "Model: Mistral-7B (requires 16GB+ VRAM)"
fi

echo "Starting comparison..."
echo "Running command: $CMD"
echo ""

# Run comparison
eval "$CMD"

echo ""
echo "=============================================="
echo "Comparison complete!"
echo "=============================================="
echo ""
echo "Results saved to: output/$OUTPUT"
echo ""
echo "To view results:"
echo "  cat output/$OUTPUT | python -m json.tool"

