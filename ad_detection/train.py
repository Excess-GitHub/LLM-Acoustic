#!/usr/bin/env python3
"""
Main training script for AD detection.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --small-gpu
    python train.py --config config.yaml --use-custom-acoustic
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.trainer import ADDetectionTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Train AD detection model using LLM + Acoustic features'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--small-gpu',
        action='store_true',
        help='Use optimizations for small GPU (e.g., GTX 1660 Ti)'
    )
    parser.add_argument(
        '--use-custom-acoustic',
        action='store_true',
        help='Use custom acoustic features from audio_feature_extraction'
    )
    parser.add_argument(
        '--n-folds', '-f',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--train-final',
        action='store_true',
        help='Train final model on all data (no CV)'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override output dir if specified
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    
    # Create trainer
    trainer = ADDetectionTrainer(
        config=config,
        small_gpu=args.small_gpu,
        use_custom_acoustic=args.use_custom_acoustic,
        output_dir=args.output_dir
    )
    
    # Load data
    trainer.load_data()
    
    if args.train_final:
        # Train final model
        results = trainer.train_final_model()
        print(f"\nFinal model trained with {results['n_samples']} samples")
        print(f"Feature dimension: {results['feature_dim']}")
        print(f"Saved to: {results['model_path']}")
    else:
        # Run cross-validation
        results = trainer.run_cross_validation(n_folds=args.n_folds)
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        for metric, stats in results['summary'].items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == '__main__':
    main()

