#!/usr/bin/env python3
"""
Compare performance with different acoustic feature sources.

This script compares:
1. VGGish + GRU-AE (paper method)
2. Custom acoustic features (from audio_feature_extraction pipeline)
3. LLM-only (no acoustic features)

Usage:
    python compare_features.py --config config.yaml
    python compare_features.py --config config.yaml --small-gpu
"""

import argparse
import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_gpu_memory(llm_extractor=None):
    """Clear GPU memory, optionally unloading LLM model."""
    import gc
    
    if llm_extractor is not None:
        # Delete model references
        if hasattr(llm_extractor, 'model') and llm_extractor.model is not None:
            del llm_extractor.model
            llm_extractor.model = None
        if hasattr(llm_extractor, 'peft_model') and llm_extractor.peft_model is not None:
            del llm_extractor.peft_model
            llm_extractor.peft_model = None
        if hasattr(llm_extractor, 'tokenizer') and llm_extractor.tokenizer is not None:
            del llm_extractor.tokenizer
            llm_extractor.tokenizer = None
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU memory cleared: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

from src.trainer import ADDetectionTrainer
from src.data_loader import PittCorpusDataset
from src.llm_features import create_llm_extractor
from src.vggish_features import VGGishFeatureExtractor
from src.gru_autoencoder import create_gru_autoencoder, GRUAutoencoderTrainer, VGGishDataset
from src.feature_fusion import FeatureFusion, CustomAcousticFusion, LLMOnlyFusion
from src.classifier import ADClassifier, create_classifier

from torch.utils.data import DataLoader


def load_or_finetune_llm(
    config: dict,
    llm_extractor,
    train_transcripts: list,
    train_labels: list,
    fold_idx: int,
    small_gpu: bool = False
):
    """Load existing checkpoint or fine-tune LLM."""
    if llm_extractor.is_fine_tuned:
        return
    
    output_dir = config['paths'].get('output_dir', './output')
    
    # Check for existing checkpoint (from train.py runs)
    # Use fold-specific checkpoint paths to match trainer.py structure
    checkpoint_dirs = [
        # New fold-specific paths
        os.path.join(output_dir, 'small_gpu', 'llm_checkpoints', f'fold_{fold_idx}', 'final'),
        os.path.join(output_dir, 'llm_checkpoints', f'fold_{fold_idx}', 'final'),
        os.path.join(output_dir, 'normal_gpu', 'llm_checkpoints', f'fold_{fold_idx}', 'final'),
        # Legacy paths (for backwards compatibility)
        os.path.join(output_dir, 'small_gpu', 'llm_checkpoints', 'final'),
        os.path.join(output_dir, 'llm_checkpoints', 'final'),
        os.path.join(output_dir, 'normal_gpu', 'llm_checkpoints', 'final'),
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir) and os.path.exists(os.path.join(checkpoint_dir, 'adapter_config.json')):
            print(f"Found existing fine-tuned model at {checkpoint_dir}")
            print("Loading saved checkpoint (skipping fine-tuning)...")
            try:
                llm_extractor.load(checkpoint_dir)
                print("Successfully loaded fine-tuned model!")
                return
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                break
    
    # No checkpoint found, fine-tune
    print("No checkpoint found, fine-tuning from scratch...")
    train_config = config.get('training', {})
    
    if small_gpu:
        small_config = config.get('small_gpu', {})
        batch_size = small_config.get('batch_size', 1)
        grad_accum = small_config.get('gradient_accumulation_steps', 16)
    else:
        batch_size = train_config.get('batch_size', 2)
        grad_accum = train_config.get('gradient_accumulation_steps', 8)
    
    # Use fold-specific checkpoint directory to match trainer.py structure
    checkpoint_dir = os.path.join(output_dir, 'llm_checkpoints', f'fold_{fold_idx}')
    
    llm_extractor.fine_tune(
        train_transcripts=train_transcripts,
        train_labels=train_labels,
        epochs=train_config.get('epochs', 5),
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        output_dir=checkpoint_dir,
        gradient_checkpointing=small_gpu
    )


def run_llm_only_experiment(
    config: dict,
    train_samples,
    test_samples,
    llm_extractor,
    fold_idx: int,
    small_gpu: bool = False
) -> dict:
    """Run experiment with LLM features only."""
    print(f"\n[LLM-Only] Fold {fold_idx + 1}")
    
    train_labels = [1 if s.group == 'AD' else 0 for s in train_samples]
    test_labels = [1 if s.group == 'AD' else 0 for s in test_samples]
    
    # Get transcripts
    train_transcripts = [s.transcript_text or '' for s in train_samples]
    test_transcripts = [s.transcript_text or '' for s in test_samples]
    
    # Load checkpoint or fine-tune
    load_or_finetune_llm(config, llm_extractor, train_transcripts, train_labels, fold_idx, small_gpu)
    
    # Extract features
    train_features = llm_extractor.extract_features(train_transcripts)
    test_features = llm_extractor.extract_features(test_transcripts)
    
    # Feature selection
    fusion = LLMOnlyFusion(importance_threshold=0.95)
    train_selected = fusion.fit_transform(train_features, np.array(train_labels))
    test_selected = fusion.transform(test_features)
    
    # Train classifier
    classifier = create_classifier(config)
    classifier.fit(train_selected, np.array(train_labels))
    
    # Evaluate
    results = classifier.evaluate(test_selected, np.array(test_labels))
    
    return results


def run_vggish_experiment(
    config: dict,
    train_samples,
    test_samples,
    llm_extractor,
    fold_idx: int,
    small_gpu: bool = False
) -> dict:
    """Run experiment with VGGish + GRU-AE (paper method)."""
    print(f"\n[VGGish + GRU-AE] Fold {fold_idx + 1}")
    
    train_labels = [1 if s.group == 'AD' else 0 for s in train_samples]
    test_labels = [1 if s.group == 'AD' else 0 for s in test_samples]
    
    # Get transcripts
    train_transcripts = [s.transcript_text or '' for s in train_samples]
    test_transcripts = [s.transcript_text or '' for s in test_samples]
    
    # Load checkpoint or fine-tune (reuses if already loaded)
    load_or_finetune_llm(config, llm_extractor, train_transcripts, train_labels, fold_idx, small_gpu)
    
    # Extract LLM features
    train_llm = llm_extractor.extract_features(train_transcripts)
    test_llm = llm_extractor.extract_features(test_transcripts)
    
    # Clear GPU memory before loading VGGish (critical for small GPUs)
    print("Clearing GPU memory before VGGish extraction...")
    clear_gpu_memory(llm_extractor)
    
    # Extract VGGish features
    vggish_extractor = VGGishFeatureExtractor()
    train_vggish = vggish_extractor.extract_batch([s.audio_path for s in train_samples])
    test_vggish = vggish_extractor.extract_batch([s.audio_path for s in test_samples])
    
    # Train GRU-AE
    gru_autoencoder = create_gru_autoencoder(config)
    dataset = VGGishDataset(train_vggish, train_labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = config.get('training', {}).get('device', 'cuda')
    trainer = GRUAutoencoderTrainer(
        model=gru_autoencoder,
        reconstruction_weight=0.5,
        learning_rate=0.001,
        device=device
    )
    trainer.train(train_loader=train_loader, epochs=50, patience=10)
    
    # Extract acoustic features
    train_acoustic = trainer.extract_features(train_vggish)
    test_acoustic = trainer.extract_features(test_vggish)
    
    # Fuse features
    fusion = FeatureFusion(importance_threshold=0.95)
    train_fused = fusion.fit_transform(train_llm, train_acoustic, np.array(train_labels))
    test_fused = fusion.transform(test_llm, test_acoustic)
    
    # Train classifier
    classifier = create_classifier(config)
    classifier.fit(train_fused, np.array(train_labels))
    
    # Evaluate
    results = classifier.evaluate(test_fused, np.array(test_labels))
    
    return results


def run_custom_acoustic_experiment(
    config: dict,
    dataset: PittCorpusDataset,
    train_samples,
    test_samples,
    llm_extractor,
    fold_idx: int,
    small_gpu: bool = False
) -> dict:
    """Run experiment with custom acoustic features."""
    print(f"\n[Custom Acoustic] Fold {fold_idx + 1}")
    
    train_labels = [1 if s.group == 'AD' else 0 for s in train_samples]
    test_labels = [1 if s.group == 'AD' else 0 for s in test_samples]
    
    # Get transcripts
    train_transcripts = [s.transcript_text or '' for s in train_samples]
    test_transcripts = [s.transcript_text or '' for s in test_samples]
    
    # Reload LLM if it was cleared after VGGish experiment
    if llm_extractor.model is None:
        print("Reloading LLM model (was cleared for VGGish)...")
        llm_extractor.load_model()
        llm_extractor.is_fine_tuned = False  # Reset so checkpoint will be loaded
    
    # Load checkpoint or fine-tune (reuses if already loaded)
    load_or_finetune_llm(config, llm_extractor, train_transcripts, train_labels, fold_idx, small_gpu)
    
    # Extract LLM features
    train_llm = llm_extractor.extract_features(train_transcripts)
    test_llm = llm_extractor.extract_features(test_transcripts)
    
    # Get custom acoustic features
    train_acoustic = []
    for sample in train_samples:
        feat = dataset.get_custom_features(sample)
        if feat is not None:
            train_acoustic.append(feat)
        else:
            train_acoustic.append(np.zeros(42))
    train_acoustic = np.vstack(train_acoustic)
    
    test_acoustic = []
    for sample in test_samples:
        feat = dataset.get_custom_features(sample)
        if feat is not None:
            test_acoustic.append(feat)
        else:
            test_acoustic.append(np.zeros(42))
    test_acoustic = np.vstack(test_acoustic)
    
    # Fuse features
    fusion = CustomAcousticFusion(importance_threshold=0.95)
    train_fused = fusion.fit_transform(train_llm, train_acoustic, np.array(train_labels))
    test_fused = fusion.transform(test_llm, test_acoustic)
    
    # Train classifier
    classifier = create_classifier(config)
    classifier.fit(train_fused, np.array(train_labels))
    
    # Evaluate
    results = classifier.evaluate(test_fused, np.array(test_labels))
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare AD detection with different acoustic feature sources'
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
        help='Use optimizations for small GPU'
    )
    parser.add_argument(
        '--n-folds', '-f',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='comparison_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    seed = config.get('dataset', {}).get('random_seed', 42)
    set_seed(seed)
    
    # Load dataset
    print("Loading dataset...")
    dataset = PittCorpusDataset(
        corpus_root=config['paths']['pitt_corpus_root'],
        custom_features_csv=config['paths'].get('custom_audio_features'),
        random_seed=seed
    )
    
    # Get folds
    folds = dataset.get_cv_folds(n_folds=args.n_folds)
    
    # Initialize LLM extractor
    print("Initializing LLM extractor...")
    llm_extractor = create_llm_extractor(config, small_gpu=args.small_gpu)
    llm_extractor.load_model()
    
    # Results storage
    all_results = {
        'llm_only': [],
        'vggish_gru': [],
        'custom_acoustic': []
    }
    
    for fold_idx, (train_samples, test_samples) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
        print(f"{'='*60}")
        
        # Don't reset - let it reuse the checkpoint if loaded
        # llm_extractor.is_fine_tuned = False
        
        # 1. LLM-only experiment
        llm_results = run_llm_only_experiment(
            config, train_samples, test_samples, llm_extractor, fold_idx, args.small_gpu
        )
        all_results['llm_only'].append(llm_results)
        print(f"  LLM-only accuracy: {llm_results['accuracy']:.4f}")
        
        # 2. VGGish + GRU-AE experiment (paper method)
        vggish_results = run_vggish_experiment(
            config, train_samples, test_samples, llm_extractor, fold_idx, args.small_gpu
        )
        all_results['vggish_gru'].append(vggish_results)
        print(f"  VGGish+GRU accuracy: {vggish_results['accuracy']:.4f}")
        
        # 3. Custom acoustic features experiment
        if dataset.custom_features_df is not None:
            custom_results = run_custom_acoustic_experiment(
                config, dataset, train_samples, test_samples, llm_extractor, fold_idx, args.small_gpu
            )
            all_results['custom_acoustic'].append(custom_results)
            print(f"  Custom acoustic accuracy: {custom_results['accuracy']:.4f}")
        else:
            print("  Custom acoustic: SKIPPED (no features file)")
    
    # Compute summary statistics
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    summary = {}
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for method, results_list in all_results.items():
        if not results_list:
            continue
            
        summary[method] = {}
        print(f"\n{method.upper()}:")
        
        for metric in metrics:
            values = [r.get(metric, 0) for r in results_list]
            mean = np.mean(values)
            std = np.std(values)
            summary[method][metric] = {'mean': mean, 'std': std}
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    
    # Statistical comparison
    print("\n" + "-"*40)
    print("PERFORMANCE IMPROVEMENT")
    print("-"*40)
    
    if 'vggish_gru' in summary and 'llm_only' in summary:
        vggish_acc = summary['vggish_gru']['accuracy']['mean']
        llm_acc = summary['llm_only']['accuracy']['mean']
        improvement = (vggish_acc - llm_acc) / llm_acc * 100
        print(f"VGGish+GRU vs LLM-only: {improvement:+.2f}%")
    
    if 'custom_acoustic' in summary and 'llm_only' in summary:
        custom_acc = summary['custom_acoustic']['accuracy']['mean']
        llm_acc = summary['llm_only']['accuracy']['mean']
        improvement = (custom_acc - llm_acc) / llm_acc * 100
        print(f"Custom Acoustic vs LLM-only: {improvement:+.2f}%")
    
    if 'custom_acoustic' in summary and 'vggish_gru' in summary:
        custom_acc = summary['custom_acoustic']['accuracy']['mean']
        vggish_acc = summary['vggish_gru']['accuracy']['mean']
        improvement = (custom_acc - vggish_acc) / vggish_acc * 100
        print(f"Custom Acoustic vs VGGish+GRU: {improvement:+.2f}%")
    
    # Save results
    output_data = {
        'config': {
            'n_folds': args.n_folds,
            'small_gpu': args.small_gpu
        },
        'fold_results': all_results,
        'summary': summary,
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = os.path.join(
        config['paths'].get('output_dir', './output'),
        args.output
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=float)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

