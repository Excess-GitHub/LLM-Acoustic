#!/usr/bin/env python3
"""
Example Usage Script for Audio Feature Extraction Pipeline.

This script demonstrates how to:
1. Extract features from a single audio file
2. Process multiple files
3. Process the entire Pitt Corpus
4. Analyze and visualize the extracted features

Run from the audio_feature_extraction directory:
    python example_usage.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# Import the main extraction functions
from main_extractor import (
    extract_audio_features,
    process_pitt_corpus,
    ExtractionConfig,
    get_all_feature_names,
    print_feature_descriptions
)
from utils.audio_loader import load_audio, get_audio_info
from utils.pitt_metadata import load_pitt_corpus, get_corpus_stats
from utils.validation import validate_features, generate_qc_report


def example_single_file():
    """Example: Extract features from a single audio file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single File Feature Extraction")
    print("=" * 60)
    
    # Path to a sample audio file (adjust as needed)
    corpus_root = "../Pitt Corpus/Pitt Corpus"
    sample_audio = os.path.join(
        corpus_root, "Media/Dementia/Cookie/WAV/001-0.wav"
    )
    
    if not os.path.exists(sample_audio):
        print(f"Sample audio not found: {sample_audio}")
        print("Please adjust the path or run example_batch_processing() instead.")
        return None
    
    # Get audio info first
    info = get_audio_info(sample_audio)
    print(f"\nAudio Info:")
    print(f"  Duration: {info['duration_seconds']:.2f} seconds")
    print(f"  Sample rate: {info['sample_rate']} Hz")
    
    # Extract features
    print("\nExtracting features...")
    features = extract_audio_features(
        audio_path=sample_audio,
        participant_id="001",
        group="AD",
        age=57,
        mmse=18,
        gender="M"
    )
    
    # Print results
    print("\nExtracted Features:")
    print("-" * 40)
    
    # VAD features
    print("\nVAD/Timing Features:")
    vad_features = ['pause_ratio', 'speech_ratio', 'pause_dur_mean', 'speech_dur_mean']
    for feat in vad_features:
        if feat in features:
            print(f"  {feat}: {features[feat]:.4f}")
    
    # Prosody features
    print("\nProsody Features:")
    prosody_features = ['f0_mean', 'f0_std', 'f0_range', 'intensity_mean']
    for feat in prosody_features:
        if feat in features:
            print(f"  {feat}: {features[feat]:.4f}")
    
    # Voice quality features
    print("\nVoice Quality Features:")
    vq_features = ['jitter_mean', 'shimmer_mean', 'hnr_mean']
    for feat in vq_features:
        if feat in features:
            print(f"  {feat}: {features[feat]:.4f}")
    
    # Validate
    is_valid, issues = validate_features(features)
    print(f"\nValidation: {'PASSED' if is_valid else 'ISSUES FOUND'}")
    if issues:
        for issue in issues[:3]:
            print(f"  - {issue}")
    
    return features


def example_corpus_stats():
    """Example: Get Pitt Corpus statistics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Pitt Corpus Statistics")
    print("=" * 60)
    
    corpus_root = "../Pitt Corpus/Pitt Corpus"
    
    if not os.path.exists(corpus_root):
        print(f"Corpus not found: {corpus_root}")
        return
    
    stats = get_corpus_stats(corpus_root, task='cookie')
    
    print(f"\nCorpus Statistics (Cookie Theft task):")
    print(f"  Total audio files: {stats['total_files']}")
    print(f"  Dementia files: {stats['dementia_files']}")
    print(f"  Control files: {stats['control_files']}")
    print(f"  Unique dementia participants: {stats['unique_participants_dementia']}")
    print(f"  Unique control participants: {stats['unique_participants_control']}")
    print(f"  Files with metadata: {stats['files_with_metadata']}")
    
    if stats['ages']:
        print(f"\n  Age range: {min(stats['ages'])} - {max(stats['ages'])} years")
        print(f"  Mean age: {np.mean(stats['ages']):.1f} years")
    
    if stats['mmse_scores']:
        print(f"  MMSE range: {min(stats['mmse_scores'])} - {max(stats['mmse_scores'])}")
        print(f"  Mean MMSE: {np.mean(stats['mmse_scores']):.1f}")


def example_batch_processing():
    """Example: Process multiple files from the corpus."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing (First 10 Files)")
    print("=" * 60)
    
    corpus_root = "../Pitt Corpus/Pitt Corpus"
    output_csv = "../output/example_features.csv"
    
    if not os.path.exists(corpus_root):
        print(f"Corpus not found: {corpus_root}")
        return None
    
    # Create output directory
    os.makedirs("../output", exist_ok=True)
    
    # Process with limited files for demonstration
    print("\nProcessing first 10 files...")
    
    df = process_pitt_corpus(
        corpus_root=corpus_root,
        output_csv=output_csv,
        task='cookie',
        max_files=10  # Limit for demo
    )
    
    print(f"\nResults saved to: {output_csv}")
    print(f"Shape: {df.shape}")
    
    # Show summary by group
    if 'group' in df.columns:
        print("\nMean features by group:")
        key_features = ['pause_ratio', 'f0_mean', 'hnr_mean']
        existing = [f for f in key_features if f in df.columns]
        if existing:
            print(df.groupby('group')[existing].mean().round(3))
    
    return df


def example_full_corpus():
    """Example: Process the entire Pitt Corpus."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Full Corpus Processing")
    print("=" * 60)
    
    corpus_root = "../Pitt Corpus/Pitt Corpus"
    output_csv = "../output/pitt_audio_features.csv"
    
    if not os.path.exists(corpus_root):
        print(f"Corpus not found: {corpus_root}")
        return None
    
    print("\nThis will process ALL files in the corpus.")
    print("Estimated time: 5-15 minutes depending on system.")
    
    response = input("\nProceed? [y/N]: ").strip().lower()
    if response != 'y':
        print("Skipped.")
        return None
    
    # Create output directory
    os.makedirs("../output", exist_ok=True)
    
    # Use custom config
    config = ExtractionConfig(
        target_sample_rate=16000,
        f0_method='pyin',
        use_parselmouth=True,
        skip_on_error=True,
        verbose=True
    )
    
    # Process full corpus
    df = process_pitt_corpus(
        corpus_root=corpus_root,
        output_csv=output_csv,
        task='cookie',
        config=config
    )
    
    print(f"\nFull results saved to: {output_csv}")
    
    # Generate QC report
    qc_report_path = output_csv.replace('.csv', '_qc_report.txt')
    generate_qc_report(df, qc_report_path)
    print(f"QC report saved to: {qc_report_path}")
    
    return df


def example_analyze_results():
    """Example: Analyze previously extracted features."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Analyze Extracted Features")
    print("=" * 60)
    
    csv_path = "../output/pitt_audio_features.csv"
    
    if not os.path.exists(csv_path):
        print(f"Features file not found: {csv_path}")
        print("Run example_batch_processing() or example_full_corpus() first.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} samples with {len(df.columns)} columns")
    
    # Group statistics
    if 'group' in df.columns:
        print("\n--- Group Comparison ---")
        
        # Key discriminative features
        key_features = [
            'pause_ratio', 'pause_dur_mean', 'speech_ratio',
            'f0_mean', 'f0_range',
            'jitter_mean', 'hnr_mean'
        ]
        existing = [f for f in key_features if f in df.columns]
        
        for feat in existing:
            ad_mean = df[df['group'] == 'AD'][feat].mean()
            hc_mean = df[df['group'] == 'HC'][feat].mean()
            
            # Simple effect size (Cohen's d approximation)
            pooled_std = df[feat].std()
            if pooled_std > 0:
                effect = (ad_mean - hc_mean) / pooled_std
            else:
                effect = 0
            
            print(f"\n{feat}:")
            print(f"  AD mean: {ad_mean:.4f}")
            print(f"  HC mean: {hc_mean:.4f}")
            print(f"  Effect size (d): {effect:+.2f}")
    
    # Correlation with MMSE
    if 'mmse' in df.columns and df['mmse'].notna().sum() > 10:
        print("\n--- Correlation with MMSE ---")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['age', 'mmse', 'duration_seconds']]
        
        correlations = []
        for col in feature_cols:
            valid = df[[col, 'mmse']].dropna()
            if len(valid) > 5:
                corr = valid[col].corr(valid['mmse'])
                correlations.append((col, corr))
        
        # Top correlations
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print("\nTop 5 features correlated with MMSE:")
        for feat, corr in correlations[:5]:
            print(f"  {feat}: r = {corr:+.3f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("AUDIO FEATURE EXTRACTION EXAMPLES")
    print("=" * 60)
    
    # Print feature descriptions
    print("\nAvailable features:")
    print_feature_descriptions()
    
    # Run examples
    print("\nSelect an example to run:")
    print("  1. Single file extraction")
    print("  2. Corpus statistics")
    print("  3. Batch processing (10 files)")
    print("  4. Full corpus processing")
    print("  5. Analyze existing results")
    print("  6. Run all (except full corpus)")
    print("  0. Exit")
    
    choice = input("\nEnter choice [1-6]: ").strip()
    
    if choice == '1':
        example_single_file()
    elif choice == '2':
        example_corpus_stats()
    elif choice == '3':
        example_batch_processing()
    elif choice == '4':
        example_full_corpus()
    elif choice == '5':
        example_analyze_results()
    elif choice == '6':
        example_single_file()
        example_corpus_stats()
        example_batch_processing()
        example_analyze_results()
    elif choice == '0':
        print("Exiting.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()

