"""
Main Audio Feature Extraction Pipeline for Pitt Corpus.

This module provides the main entry point for extracting clinician-interpretable
audio features from the Pitt Corpus dataset for Alzheimer's disease detection.

Based on the cross-linguistic study's implementation (Timing_VAD.py).

Features extracted:
1. VAD-based timing features (17 features)
2. Prosodic features - F0 and intensity (11 features)
3. Voice quality features - jitter, shimmer, HNR (8 features)

Total: ~36-42 audio features per participant
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports - handle both package and standalone usage
try:
    from .vad_features import extract_vad_features, get_vad_feature_names
    from .prosody_features import extract_prosody_features, get_prosody_feature_names
    from .voice_quality_features import extract_voice_quality_features, get_voice_quality_feature_names
    from .utils.audio_loader import load_audio, preprocess_audio
    from .utils.pitt_metadata import load_pitt_corpus, get_corpus_stats, ParticipantInfo
    from .utils.validation import (
        validate_features, 
        validate_audio_file, 
        generate_qc_report,
        flag_problematic_samples
    )
except ImportError:
    from vad_features import extract_vad_features, get_vad_feature_names
    from prosody_features import extract_prosody_features, get_prosody_feature_names
    from voice_quality_features import extract_voice_quality_features, get_voice_quality_feature_names
    from utils.audio_loader import load_audio, preprocess_audio
    from utils.pitt_metadata import load_pitt_corpus, get_corpus_stats, ParticipantInfo
    from utils.validation import (
        validate_features, 
        validate_audio_file, 
        generate_qc_report,
        flag_problematic_samples
    )

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for audio feature extraction."""
    # Audio preprocessing
    target_sample_rate: int = 16000
    normalize: bool = True
    
    # VAD parameters
    vad_window: float = 0.025  # 25ms
    vad_hop: float = 0.01  # 10ms
    
    # Prosody parameters
    f0_min: float = 50.0
    f0_max: float = 500.0
    f0_method: str = 'pyin'
    frame_length: int = 2048
    hop_length: int = 512
    
    # Voice quality parameters
    use_parselmouth: bool = True
    hnr_time_step: float = 0.01
    hnr_min_pitch: float = 75.0
    hnr_silence_threshold: float = 0.1
    
    # Feature extraction flags
    compute_vad: bool = True
    compute_prosody: bool = True
    compute_voice_quality: bool = True
    
    # Processing options
    skip_on_error: bool = True
    verbose: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExtractionConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            target_sample_rate=config_dict.get('audio', {}).get('target_sample_rate', 16000),
            normalize=config_dict.get('audio', {}).get('normalize', True),
            vad_window=config_dict.get('vad', {}).get('window_size', 0.025),
            vad_hop=config_dict.get('vad', {}).get('hop_size', 0.01),
            f0_min=config_dict.get('prosody', {}).get('f0_min', 50.0),
            f0_max=config_dict.get('prosody', {}).get('f0_max', 500.0),
            f0_method=config_dict.get('prosody', {}).get('f0_method', 'pyin'),
            frame_length=config_dict.get('prosody', {}).get('frame_length', 2048),
            hop_length=config_dict.get('prosody', {}).get('hop_length', 512),
            use_parselmouth=True,  # Always try parselmouth first
            hnr_time_step=config_dict.get('voice_quality', {}).get('hnr_time_step', 0.01),
            hnr_min_pitch=config_dict.get('voice_quality', {}).get('hnr_min_pitch', 75.0),
            hnr_silence_threshold=config_dict.get('voice_quality', {}).get('hnr_silence_threshold', 0.1),
            compute_vad=config_dict.get('features', {}).get('compute_vad', True),
            compute_prosody=config_dict.get('features', {}).get('compute_prosody', True),
            compute_voice_quality=config_dict.get('features', {}).get('compute_voice_quality', True),
            skip_on_error=config_dict.get('processing', {}).get('skip_on_error', True),
            verbose=config_dict.get('processing', {}).get('verbose', True)
        )


def extract_audio_features(
    audio_path: str,
    participant_id: str = "",
    group: str = "",
    age: Optional[int] = None,
    mmse: Optional[int] = None,
    gender: Optional[str] = None,
    config: Optional[ExtractionConfig] = None
) -> Dict[str, Any]:
    """
    Extract all audio features from a single audio file.
    
    This is the main function for extracting ~42 clinician-interpretable
    audio features from a speech recording.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file (WAV or MP3)
    participant_id : str
        Participant identifier
    group : str
        Group label ('AD' for dementia, 'HC' for healthy control)
    age : int, optional
        Participant age
    mmse : int, optional
        Mini-Mental State Examination score
    gender : str, optional
        Participant gender ('M' or 'F')
    config : ExtractionConfig, optional
        Extraction configuration (uses defaults if not provided)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - Metadata: participant_id, group, age, mmse
        - VAD features: pause_ratio, speech_ratio, etc.
        - Prosody features: f0_mean, f0_std, intensity_mean, etc.
        - Voice quality features: jitter_mean, shimmer_mean, hnr_mean, etc.
    """
    if config is None:
        config = ExtractionConfig()
    
    features = {
        'participant_id': participant_id,
        'group': group,
        'age': age,
        'mmse': mmse,
        'gender': gender,
        'audio_path': audio_path
    }
    
    try:
        # Load and preprocess audio
        sig, fs = load_audio(
            audio_path,
            target_sr=config.target_sample_rate,
            normalize=config.normalize
        )
        
        duration = len(sig) / fs
        features['duration_seconds'] = duration
        
        if duration < 1.0:
            logger.warning(f"Very short audio ({duration:.2f}s): {audio_path}")
        
        # Extract VAD features
        if config.compute_vad:
            try:
                vad_feats = extract_vad_features(
                    sig, fs,
                    win=config.vad_window,
                    step=config.vad_hop
                )
                features.update(vad_feats)
            except Exception as e:
                logger.error(f"VAD extraction failed for {audio_path}: {e}")
                # Fill with zeros
                for name in get_vad_feature_names():
                    features[name] = 0.0
        
        # Extract prosody features
        if config.compute_prosody:
            try:
                prosody_feats = extract_prosody_features(
                    sig, fs,
                    f0_method=config.f0_method,
                    f0_min=config.f0_min,
                    f0_max=config.f0_max,
                    frame_length=config.frame_length,
                    hop_length=config.hop_length
                )
                features.update(prosody_feats)
            except Exception as e:
                logger.error(f"Prosody extraction failed for {audio_path}: {e}")
                for name in get_prosody_feature_names():
                    features[name] = 0.0
        
        # Extract voice quality features
        if config.compute_voice_quality:
            try:
                vq_feats = extract_voice_quality_features(
                    sig, fs,
                    f0_min=config.f0_min,
                    f0_max=config.f0_max,
                    use_parselmouth=config.use_parselmouth,
                    hnr_time_step=config.hnr_time_step,
                    hnr_min_pitch=config.hnr_min_pitch,
                    hnr_silence_threshold=config.hnr_silence_threshold
                )
                features.update(vq_feats)
            except Exception as e:
                logger.error(f"Voice quality extraction failed for {audio_path}: {e}")
                for name in get_voice_quality_feature_names():
                    features[name] = 0.0
        
        features['extraction_success'] = True
        
    except Exception as e:
        logger.error(f"Feature extraction failed for {audio_path}: {e}")
        features['extraction_success'] = False
        features['error'] = str(e)
    
    return features


def process_pitt_corpus(
    corpus_root: str,
    output_csv: str,
    task: str = 'cookie',
    config: Optional[ExtractionConfig] = None,
    max_files: Optional[int] = None
) -> pd.DataFrame:
    """
    Process entire Pitt Corpus and extract features.
    
    Parameters
    ----------
    corpus_root : str
        Path to Pitt Corpus root directory
    output_csv : str
        Path for output CSV file
    task : str
        Task to process ('cookie' for Cookie Theft description)
    config : ExtractionConfig, optional
        Extraction configuration
    max_files : int, optional
        Maximum number of files to process (for testing)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all extracted features
    """
    if config is None:
        config = ExtractionConfig()
    
    logger.info(f"Processing Pitt Corpus from: {corpus_root}")
    logger.info(f"Task: {task}")
    
    # Get corpus statistics
    try:
        stats = get_corpus_stats(corpus_root, task=task)
        logger.info(f"Found {stats['total_files']} audio files")
        logger.info(f"  - Dementia: {stats['dementia_files']} ({stats['unique_participants_dementia']} participants)")
        logger.info(f"  - Control: {stats['control_files']} ({stats['unique_participants_control']} participants)")
    except Exception as e:
        logger.warning(f"Could not get corpus stats: {e}")
    
    # Collect results
    results = []
    errors = []
    
    # Create progress bar
    corpus_iter = load_pitt_corpus(corpus_root, task=task)
    if max_files:
        corpus_iter = list(corpus_iter)[:max_files]
    
    for item in tqdm(corpus_iter, desc="Extracting features"):
        audio_path = item['audio_path']
        participant_info = item['participant_info']
        
        # Extract features
        features = extract_audio_features(
            audio_path=audio_path,
            participant_id=participant_info.participant_id,
            group=participant_info.group,
            age=participant_info.age,
            mmse=participant_info.mmse,
            gender=participant_info.gender,
            config=config
        )
        
        # Add session info
        features['session'] = participant_info.session
        
        if features.get('extraction_success', False):
            results.append(features)
        else:
            errors.append({
                'audio_path': audio_path,
                'error': features.get('error', 'Unknown error')
            })
            if config.skip_on_error:
                continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    meta_cols = ['participant_id', 'session', 'group', 'age', 'mmse', 'gender', 'duration_seconds']
    vad_cols = get_vad_feature_names()
    prosody_cols = get_prosody_feature_names()
    vq_cols = get_voice_quality_feature_names()
    
    # Build column order
    column_order = []
    for col in meta_cols + vad_cols + prosody_cols + vq_cols:
        if col in df.columns:
            column_order.append(col)
    
    # Add any remaining columns
    for col in df.columns:
        if col not in column_order and col not in ['audio_path', 'extraction_success', 'error']:
            column_order.append(col)
    
    df = df[column_order]
    
    # Save to CSV
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved features to: {output_csv}")
    
    # Log summary
    logger.info(f"\nExtraction Summary:")
    logger.info(f"  - Successfully processed: {len(results)} files")
    logger.info(f"  - Errors: {len(errors)} files")
    logger.info(f"  - Total features: {len(vad_cols) + len(prosody_cols) + len(vq_cols)}")
    
    if errors and config.verbose:
        logger.info(f"\nFirst 5 errors:")
        for err in errors[:5]:
            logger.info(f"  - {err['audio_path']}: {err['error']}")
    
    return df


def get_all_feature_names() -> List[str]:
    """Return list of all feature names in order."""
    return get_vad_feature_names() + get_prosody_feature_names() + get_voice_quality_feature_names()


def print_feature_descriptions():
    """Print descriptions of all extracted features."""
    descriptions = {
        # VAD features
        'pause_ratio': 'Total pause time / total recording duration',
        'speech_ratio': 'Total speech time / total recording duration',
        'pause_speech_ratio': 'Number of speech segments / number of pauses',
        'num_pauses_per_sec': 'Number of pauses per second',
        'num_speech_segments_per_sec': 'Number of speech segments per second',
        'speech_dur_mean': 'Mean duration of speech segments (seconds)',
        'speech_dur_std': 'Standard deviation of speech segment durations',
        'speech_dur_skew': 'Skewness of speech segment duration distribution',
        'speech_dur_kurt': 'Kurtosis of speech segment duration distribution',
        'speech_dur_min': 'Minimum speech segment duration (seconds)',
        'speech_dur_max': 'Maximum speech segment duration (seconds)',
        'pause_dur_mean': 'Mean duration of pauses (seconds)',
        'pause_dur_std': 'Standard deviation of pause durations',
        'pause_dur_skew': 'Skewness of pause duration distribution',
        'pause_dur_kurt': 'Kurtosis of pause duration distribution',
        'pause_dur_min': 'Minimum pause duration (seconds)',
        'pause_dur_max': 'Maximum pause duration (seconds)',
        
        # Prosody features
        'f0_mean': 'Mean fundamental frequency (Hz) - pitch average',
        'f0_std': 'Standard deviation of F0 - pitch variability',
        'f0_min': 'Minimum F0 (Hz)',
        'f0_max': 'Maximum F0 (Hz)',
        'f0_range': 'F0 range (max - min Hz) - pitch range',
        'intensity_mean': 'Mean intensity (dB) - loudness average',
        'intensity_std': 'Standard deviation of intensity - loudness variability',
        'intensity_min': 'Minimum intensity (dB)',
        'intensity_max': 'Maximum intensity (dB)',
        'intensity_range': 'Intensity range (dB)',
        'voice_rate': 'Number of voiced segments per second',
        
        # Voice quality features
        'jitter_mean': 'Mean cycle-to-cycle F0 perturbation (relative)',
        'jitter_std': 'Standard deviation of jitter',
        'shimmer_mean': 'Mean cycle-to-cycle amplitude perturbation (relative)',
        'shimmer_std': 'Standard deviation of shimmer',
        'hnr_mean': 'Mean harmonics-to-noise ratio (dB) - voice clarity',
        'hnr_std': 'Standard deviation of HNR',
        'hnr_min': 'Minimum HNR (dB)',
        'hnr_max': 'Maximum HNR (dB)',
    }
    
    print("\n" + "=" * 70)
    print("AUDIO FEATURES FOR ALZHEIMER'S DISEASE DETECTION")
    print("=" * 70)
    
    print("\n--- VAD-BASED TIMING FEATURES ---")
    print("Clinical relevance: Longer pauses and higher pause ratios indicate")
    print("slower word retrieval and semantic memory decline in AD.\n")
    
    for name in get_vad_feature_names():
        print(f"  {name}: {descriptions.get(name, 'No description')}")
    
    print("\n--- PROSODIC FEATURES (F0 & INTENSITY) ---")
    print("Clinical relevance: Reduced F0 range and flatter intensity indicate")
    print("monotonous speech and reduced prosody in AD.\n")
    
    for name in get_prosody_feature_names():
        print(f"  {name}: {descriptions.get(name, 'No description')}")
    
    print("\n--- VOICE QUALITY FEATURES ---")
    print("Clinical relevance: Increased jitter/shimmer and lower HNR indicate")
    print("vocal instability (can overlap with normal aging).\n")
    
    for name in get_voice_quality_feature_names():
        print(f"  {name}: {descriptions.get(name, 'No description')}")
    
    print("\n" + "=" * 70)
    print(f"Total features: {len(get_all_feature_names())}")
    print("=" * 70 + "\n")


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract audio features from Pitt Corpus for AD detection"
    )
    parser.add_argument(
        '--corpus-root',
        type=str,
        default='../Pitt Corpus/Pitt Corpus',
        help='Path to Pitt Corpus root directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../output/pitt_audio_features.csv',
        help='Output CSV path'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='cookie',
        choices=['cookie', 'fluency', 'recall', 'sentence'],
        help='Task to process'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum files to process (for testing)'
    )
    parser.add_argument(
        '--list-features',
        action='store_true',
        help='Print feature descriptions and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_features:
        print_feature_descriptions()
    else:
        # Load config if available
        config = None
        if os.path.exists(args.config):
            config = ExtractionConfig.from_yaml(args.config)
        
        # Process corpus
        df = process_pitt_corpus(
            corpus_root=args.corpus_root,
            output_csv=args.output,
            task=args.task,
            config=config,
            max_files=args.max_files
        )
        
        # Generate QC report
        qc_path = args.output.replace('.csv', '_qc_report.txt')
        generate_qc_report(df, qc_path)

