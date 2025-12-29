"""
Validation and quality control utilities for audio features.

Provides:
- Feature range validation
- NaN/Inf detection
- Outlier flagging
- QC report generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Expected feature ranges based on published literature
FEATURE_RANGES = {
    # VAD features
    'pause_ratio': (0.0, 1.0),
    'speech_ratio': (0.0, 1.0),
    'pause_speech_ratio': (0.0, 100.0),  # Can be high for very few pauses
    'num_pauses_per_sec': (0.0, 10.0),
    'num_speech_segments_per_sec': (0.0, 10.0),
    
    # Pause duration statistics
    'pause_dur_mean': (0.0, 30.0),  # seconds
    'pause_dur_std': (0.0, 20.0),
    'pause_dur_skew': (-10.0, 10.0),
    'pause_dur_kurt': (-5.0, 50.0),
    'pause_dur_min': (0.0, 30.0),
    'pause_dur_max': (0.0, 60.0),
    
    # Speech duration statistics
    'speech_dur_mean': (0.0, 30.0),
    'speech_dur_std': (0.0, 20.0),
    'speech_dur_skew': (-10.0, 10.0),
    'speech_dur_kurt': (-5.0, 50.0),
    'speech_dur_min': (0.0, 30.0),
    'speech_dur_max': (0.0, 60.0),
    
    # Voice rate
    'voice_rate': (0.0, 20.0),  # voiced segments per second
    
    # F0 features
    'f0_mean': (50.0, 500.0),  # Hz
    'f0_std': (0.0, 200.0),
    'f0_min': (50.0, 500.0),
    'f0_max': (50.0, 500.0),
    'f0_range': (0.0, 450.0),
    
    # Intensity features
    'intensity_mean': (-100.0, 100.0),  # dB
    'intensity_std': (0.0, 50.0),
    'intensity_min': (-100.0, 100.0),
    'intensity_max': (-100.0, 100.0),
    'intensity_range': (0.0, 100.0),
    
    # Jitter
    'jitter_mean': (0.0, 0.2),  # proportion
    'jitter_std': (0.0, 0.2),
    
    # Shimmer
    'shimmer_mean': (0.0, 0.5),  # proportion
    'shimmer_std': (0.0, 0.5),
    
    # HNR
    'hnr_mean': (-10.0, 40.0),  # dB
    'hnr_std': (0.0, 20.0),
    'hnr_min': (-20.0, 40.0),
    'hnr_max': (-10.0, 50.0),
}


def validate_features(
    features: Dict[str, Any],
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate extracted features against expected ranges.
    
    Parameters
    ----------
    features : Dict[str, Any]
        Dictionary of feature name -> value
    strict : bool
        If True, return False for any out-of-range values.
        If False, only flag issues but still return True.
        
    Returns
    -------
    is_valid : bool
        Whether all features pass validation
    issues : List[str]
        List of validation issue messages
    """
    issues = []
    
    for feature_name, value in features.items():
        # Skip non-numeric features
        if feature_name in ['participant_id', 'group', 'gender', 'diagnosis', 'file_path']:
            continue
            
        # Check for None/NaN/Inf
        if value is None:
            issues.append(f"{feature_name}: value is None")
            continue
            
        if isinstance(value, (int, float)):
            if np.isnan(value):
                issues.append(f"{feature_name}: value is NaN")
                continue
            if np.isinf(value):
                issues.append(f"{feature_name}: value is Inf")
                continue
        
        # Check range if defined
        if feature_name in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[feature_name]
            if not (min_val <= value <= max_val):
                issues.append(
                    f"{feature_name}: value {value:.4f} outside expected range [{min_val}, {max_val}]"
                )
    
    is_valid = len(issues) == 0 if strict else True
    
    if issues:
        logger.warning(f"Validation issues found: {len(issues)}")
        for issue in issues[:5]:  # Log first 5 issues
            logger.warning(f"  - {issue}")
        if len(issues) > 5:
            logger.warning(f"  ... and {len(issues) - 5} more issues")
    
    return is_valid, issues


def validate_audio_file(
    duration: float,
    sample_rate: int,
    signal_stats: Dict[str, float],
    min_duration: float = 1.0,
    max_duration: float = 600.0
) -> Tuple[bool, List[str]]:
    """
    Validate audio file properties.
    
    Parameters
    ----------
    duration : float
        Audio duration in seconds
    sample_rate : int
        Sample rate in Hz
    signal_stats : Dict[str, float]
        Statistics about the signal (max_amplitude, rms, etc.)
    min_duration : float
        Minimum acceptable duration
    max_duration : float
        Maximum acceptable duration
        
    Returns
    -------
    is_valid : bool
        Whether audio passes validation
    issues : List[str]
        List of validation issues
    """
    issues = []
    
    # Check duration
    if duration < min_duration:
        issues.append(f"Duration {duration:.2f}s is below minimum {min_duration}s")
    if duration > max_duration:
        issues.append(f"Duration {duration:.2f}s exceeds maximum {max_duration}s")
    
    # Check sample rate
    if sample_rate not in [8000, 16000, 22050, 44100, 48000]:
        issues.append(f"Unusual sample rate: {sample_rate} Hz")
    
    # Check signal quality
    if 'max_amplitude' in signal_stats:
        if signal_stats['max_amplitude'] < 0.01:
            issues.append("Very low maximum amplitude - possible silent recording")
        if signal_stats['max_amplitude'] > 1.5:
            issues.append("Clipping detected - maximum amplitude exceeds 1.0")
    
    if 'rms' in signal_stats:
        if signal_stats['rms'] < 0.001:
            issues.append("Very low RMS - possible silent or corrupted recording")
    
    return len(issues) == 0, issues


def detect_outliers(
    df: pd.DataFrame,
    feature_columns: List[str],
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers in feature values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features
    feature_columns : List[str]
        Columns to check for outliers
    method : str
        Outlier detection method ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection
        - IQR: multiplier for IQR (default 1.5)
        - Z-score: number of standard deviations (default 3)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with boolean outlier flags for each feature
    """
    outlier_flags = pd.DataFrame(index=df.index)
    
    for col in feature_columns:
        if col not in df.columns:
            continue
            
        values = df[col].dropna()
        if len(values) < 3:
            outlier_flags[f'{col}_outlier'] = False
            continue
        
        if method == 'iqr':
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_flags[f'{col}_outlier'] = ~df[col].between(lower_bound, upper_bound)
            
        elif method == 'zscore':
            mean = values.mean()
            std = values.std()
            if std > 0:
                zscores = (df[col] - mean) / std
                outlier_flags[f'{col}_outlier'] = zscores.abs() > threshold
            else:
                outlier_flags[f'{col}_outlier'] = False
    
    return outlier_flags


def generate_qc_report(
    df: pd.DataFrame,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a quality control report for extracted features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with extracted features
    output_path : str, optional
        Path to save report (as text file)
        
    Returns
    -------
    Dict[str, Any]
        QC report as dictionary
    """
    report = {
        'summary': {},
        'feature_stats': {},
        'validation_issues': {},
        'group_comparison': {}
    }
    
    # Basic summary
    report['summary']['total_samples'] = len(df)
    report['summary']['features_count'] = len([c for c in df.columns if c not in 
                                               ['participant_id', 'session', 'group', 'age', 'mmse', 'gender']])
    
    # Count by group
    if 'group' in df.columns:
        report['summary']['by_group'] = df['group'].value_counts().to_dict()
    
    # Missing values
    report['summary']['missing_values'] = df.isnull().sum().to_dict()
    report['summary']['rows_with_missing'] = df.isnull().any(axis=1).sum()
    
    # Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['age', 'mmse']:
            continue
        report['feature_stats'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'missing': int(df[col].isnull().sum()),
            'zeros': int((df[col] == 0).sum())
        }
    
    # Validate each row
    issues_per_sample = []
    for idx, row in df.iterrows():
        features_dict = row.to_dict()
        _, issues = validate_features(features_dict)
        if issues:
            issues_per_sample.append({
                'index': idx,
                'participant_id': row.get('participant_id', 'unknown'),
                'issues': issues
            })
    
    report['validation_issues']['samples_with_issues'] = len(issues_per_sample)
    report['validation_issues']['details'] = issues_per_sample[:10]  # First 10
    
    # Group comparison (if group column exists)
    if 'group' in df.columns:
        for col in numeric_cols:
            if col in ['age', 'mmse']:
                continue
            group_stats = df.groupby('group')[col].agg(['mean', 'std', 'count']).to_dict('index')
            report['group_comparison'][col] = group_stats
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("AUDIO FEATURE EXTRACTION QC REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            for key, value in report['summary'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("VALIDATION ISSUES\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Samples with issues: {report['validation_issues']['samples_with_issues']}\n")
            for item in report['validation_issues']['details'][:5]:
                f.write(f"  - {item['participant_id']}: {len(item['issues'])} issues\n")
            f.write("\n")
            
            f.write("FEATURE STATISTICS (first 10)\n")
            f.write("-" * 40 + "\n")
            for i, (col, stats) in enumerate(report['feature_stats'].items()):
                if i >= 10:
                    f.write(f"  ... and {len(report['feature_stats']) - 10} more features\n")
                    break
                f.write(f"  {col}:\n")
                f.write(f"    mean={stats['mean']:.4f}, std={stats['std']:.4f}\n")
                f.write(f"    range=[{stats['min']:.4f}, {stats['max']:.4f}]\n")
            
        logger.info(f"QC report saved to {output_path}")
    
    return report


def flag_problematic_samples(
    df: pd.DataFrame,
    min_speech_ratio: float = 0.05,
    max_pause_ratio: float = 0.95
) -> pd.DataFrame:
    """
    Flag samples that may have processing issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features
    min_speech_ratio : float
        Minimum acceptable speech ratio
    max_pause_ratio : float
        Maximum acceptable pause ratio
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added flag columns
    """
    df = df.copy()
    
    # Initialize flags
    df['flag_low_speech'] = False
    df['flag_high_pause'] = False
    df['flag_missing_f0'] = False
    df['flag_extreme_values'] = False
    
    # Check speech/pause ratios
    if 'speech_ratio' in df.columns:
        df['flag_low_speech'] = df['speech_ratio'] < min_speech_ratio
    if 'pause_ratio' in df.columns:
        df['flag_high_pause'] = df['pause_ratio'] > max_pause_ratio
    
    # Check F0
    if 'f0_mean' in df.columns:
        df['flag_missing_f0'] = df['f0_mean'].isnull() | (df['f0_mean'] == 0)
    
    # Check for any NaN in critical features
    critical_features = ['pause_ratio', 'speech_ratio', 'f0_mean', 'intensity_mean']
    existing_critical = [f for f in critical_features if f in df.columns]
    if existing_critical:
        df['flag_missing_values'] = df[existing_critical].isnull().any(axis=1)
    
    # Summary flag
    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    df['any_flag'] = df[flag_cols].any(axis=1)
    
    flagged_count = df['any_flag'].sum()
    logger.info(f"Flagged {flagged_count} / {len(df)} samples ({100*flagged_count/len(df):.1f}%)")
    
    return df

