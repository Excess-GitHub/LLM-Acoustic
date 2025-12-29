"""
Audio Feature Extraction Pipeline for Pitt Corpus.

A complete Python pipeline for extracting clinician-interpretable audio features
from the Pitt Corpus dataset for Alzheimer's disease detection.

Based on the cross-linguistic study's implementation (Timing_VAD.py).

Features:
- VAD-based timing features (17 features)
- Prosodic features - F0 and intensity (11 features)  
- Voice quality features - jitter, shimmer, HNR (8 features)

Total: ~36 audio features per participant

Usage:
    from audio_feature_extraction import extract_audio_features, process_pitt_corpus
    
    # Single file
    features = extract_audio_features('path/to/audio.wav', participant_id='001')
    
    # Batch processing
    df = process_pitt_corpus('path/to/corpus', 'output.csv')

Author: Based on cross-linguistic AD detection study
"""

__version__ = "1.0.0"
__author__ = "Audio Feature Extraction Pipeline"

# Main extraction functions
from .main_extractor import (
    extract_audio_features,
    process_pitt_corpus,
    ExtractionConfig,
    get_all_feature_names,
    print_feature_descriptions
)

# Feature extraction modules
from .vad_features import (
    eVAD,
    duration_feats,
    extract_vad_features,
    get_vad_feature_names
)

from .prosody_features import (
    extract_f0,
    extract_intensity,
    extract_prosody_features,
    get_prosody_feature_names
)

from .voice_quality_features import (
    compute_jitter,
    compute_shimmer,
    compute_hnr,
    extract_voice_quality_features,
    get_voice_quality_feature_names
)

# Utilities
from .utils import (
    load_audio,
    preprocess_audio,
    parse_chat_file,
    load_pitt_corpus,
    validate_features,
    generate_qc_report
)

__all__ = [
    # Main functions
    'extract_audio_features',
    'process_pitt_corpus',
    'ExtractionConfig',
    'get_all_feature_names',
    'print_feature_descriptions',
    
    # VAD features
    'eVAD',
    'duration_feats',
    'extract_vad_features',
    'get_vad_feature_names',
    
    # Prosody features
    'extract_f0',
    'extract_intensity',
    'extract_prosody_features',
    'get_prosody_feature_names',
    
    # Voice quality features
    'compute_jitter',
    'compute_shimmer',
    'compute_hnr',
    'extract_voice_quality_features',
    'get_voice_quality_feature_names',
    
    # Utilities
    'load_audio',
    'preprocess_audio',
    'parse_chat_file',
    'load_pitt_corpus',
    'validate_features',
    'generate_qc_report',
]

