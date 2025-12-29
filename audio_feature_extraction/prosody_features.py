"""
Prosodic features extraction (F0 and Intensity).

Extracts:
- Fundamental Frequency (F0) features: mean, std, min, max, range
- Intensity (Energy) features: mean, std, min, max, range

Supports multiple F0 extraction methods:
- pYIN (librosa) - probabilistic YIN, more robust
- YIN (librosa) - fast, deterministic
- Parselmouth (Praat) - gold standard for speech analysis
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# Try to import parselmouth (optional but recommended)
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logger.warning("Parselmouth not available. Using librosa for F0 extraction.")

import librosa


def extract_f0_pyin(
    sig: np.ndarray,
    fs: int,
    fmin: float = 50.0,
    fmax: float = 500.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract F0 using probabilistic YIN (pYIN) algorithm.
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal
    fs : int
        Sample rate
    fmin : float
        Minimum frequency for F0 search
    fmax : float
        Maximum frequency for F0 search
    frame_length : int
        Frame length for analysis
    hop_length : int
        Hop length between frames
        
    Returns
    -------
    f0 : np.ndarray
        F0 values per frame (NaN for unvoiced)
    voiced_flag : np.ndarray
        Boolean array indicating voiced frames
    voiced_prob : np.ndarray
        Probability of voicing per frame
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        sig,
        fmin=fmin,
        fmax=fmax,
        sr=fs,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Replace NaN with 0 for unvoiced frames
    f0 = np.nan_to_num(f0, nan=0.0)
    
    return f0, voiced_flag, voiced_prob


def extract_f0_yin(
    sig: np.ndarray,
    fs: int,
    fmin: float = 50.0,
    fmax: float = 500.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract F0 using YIN algorithm (faster but less robust than pYIN).
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal
    fs : int
        Sample rate
    fmin : float
        Minimum frequency for F0 search
    fmax : float
        Maximum frequency for F0 search
    frame_length : int
        Frame length for analysis
    hop_length : int
        Hop length between frames
        
    Returns
    -------
    np.ndarray
        F0 values per frame
    """
    f0 = librosa.yin(
        sig,
        fmin=fmin,
        fmax=fmax,
        sr=fs,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    return f0


def extract_f0_parselmouth(
    sig: np.ndarray,
    fs: int,
    fmin: float = 50.0,
    fmax: float = 500.0,
    time_step: float = 0.01
) -> np.ndarray:
    """
    Extract F0 using Parselmouth (Praat).
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal
    fs : int
        Sample rate
    fmin : float
        Minimum frequency for F0 search
    fmax : float
        Maximum frequency for F0 search
    time_step : float
        Time step between F0 measurements (seconds)
        
    Returns
    -------
    np.ndarray
        F0 values per frame (0 for unvoiced)
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ImportError("Parselmouth is not available. Install with: pip install praat-parselmouth")
    
    # Create Parselmouth Sound object
    sound = parselmouth.Sound(sig, sampling_frequency=fs)
    
    # Extract pitch
    pitch = call(sound, "To Pitch", time_step, fmin, fmax)
    
    # Get F0 values
    n_frames = call(pitch, "Get number of frames")
    f0_values = []
    
    for i in range(1, n_frames + 1):
        f0 = call(pitch, "Get value in frame", i, "Hertz")
        if np.isnan(f0):
            f0_values.append(0.0)
        else:
            f0_values.append(f0)
    
    return np.array(f0_values)


def extract_f0(
    sig: np.ndarray,
    fs: int,
    method: str = 'pyin',
    fmin: float = 50.0,
    fmax: float = 500.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract F0 using specified method.
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal
    fs : int
        Sample rate
    method : str
        Extraction method: 'pyin', 'yin', or 'parselmouth'
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    frame_length : int
        Frame length (for librosa methods)
    hop_length : int
        Hop length (for librosa methods)
        
    Returns
    -------
    np.ndarray
        F0 contour
    """
    if method == 'pyin':
        f0, _, _ = extract_f0_pyin(sig, fs, fmin, fmax, frame_length, hop_length)
    elif method == 'yin':
        f0 = extract_f0_yin(sig, fs, fmin, fmax, frame_length, hop_length)
    elif method == 'parselmouth':
        if PARSELMOUTH_AVAILABLE:
            time_step = hop_length / fs
            f0 = extract_f0_parselmouth(sig, fs, fmin, fmax, time_step)
        else:
            logger.warning("Parselmouth not available, falling back to pYIN")
            f0, _, _ = extract_f0_pyin(sig, fs, fmin, fmax, frame_length, hop_length)
    else:
        raise ValueError(f"Unknown F0 method: {method}")
    
    return f0


def extract_intensity(
    sig: np.ndarray,
    fs: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    use_parselmouth: bool = False
) -> np.ndarray:
    """
    Extract intensity (energy) contour from audio.
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal
    fs : int
        Sample rate
    frame_length : int
        Frame length for analysis
    hop_length : int
        Hop length between frames
    use_parselmouth : bool
        Whether to use Parselmouth for extraction
        
    Returns
    -------
    np.ndarray
        Intensity values in dB per frame
    """
    if use_parselmouth and PARSELMOUTH_AVAILABLE:
        return _extract_intensity_parselmouth(sig, fs, hop_length / fs)
    else:
        return _extract_intensity_librosa(sig, fs, frame_length, hop_length)


def _extract_intensity_librosa(
    sig: np.ndarray,
    fs: int,
    frame_length: int,
    hop_length: int
) -> np.ndarray:
    """Extract intensity using librosa RMS energy."""
    # Compute RMS energy
    rms = librosa.feature.rms(
        y=sig,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Convert to dB
    eps = 1e-10
    intensity_db = 20 * np.log10(rms + eps)
    
    return intensity_db


def _extract_intensity_parselmouth(
    sig: np.ndarray,
    fs: int,
    time_step: float = 0.01
) -> np.ndarray:
    """Extract intensity using Parselmouth (Praat)."""
    if not PARSELMOUTH_AVAILABLE:
        raise ImportError("Parselmouth not available")
    
    sound = parselmouth.Sound(sig, sampling_frequency=fs)
    intensity = call(sound, "To Intensity", 100, time_step, "yes")
    
    n_frames = call(intensity, "Get number of frames")
    intensity_values = []
    
    for i in range(1, n_frames + 1):
        val = call(intensity, "Get value in frame", i)
        if np.isnan(val):
            intensity_values.append(-100.0)
        else:
            intensity_values.append(val)
    
    return np.array(intensity_values)


def compute_f0_statistics(
    f0: np.ndarray,
    voiced_threshold: float = 0.0
) -> Dict[str, float]:
    """
    Compute statistics on F0 contour (voiced frames only).
    
    Parameters
    ----------
    f0 : np.ndarray
        F0 values per frame (0 or NaN for unvoiced)
    voiced_threshold : float
        Threshold for considering a frame voiced
        
    Returns
    -------
    Dict[str, float]
        F0 statistics: mean, std, min, max, range
    """
    # Filter to voiced frames only
    voiced_f0 = f0[f0 > voiced_threshold]
    
    if len(voiced_f0) == 0:
        logger.warning("No voiced frames found for F0 statistics")
        return {
            'f0_mean': 0.0,
            'f0_std': 0.0,
            'f0_min': 0.0,
            'f0_max': 0.0,
            'f0_range': 0.0
        }
    
    f0_mean = float(np.mean(voiced_f0))
    f0_std = float(np.std(voiced_f0))
    f0_min = float(np.min(voiced_f0))
    f0_max = float(np.max(voiced_f0))
    f0_range = f0_max - f0_min
    
    return {
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'f0_min': f0_min,
        'f0_max': f0_max,
        'f0_range': f0_range
    }


def compute_intensity_statistics(
    intensity: np.ndarray,
    min_valid_db: float = -100.0
) -> Dict[str, float]:
    """
    Compute statistics on intensity contour.
    
    Parameters
    ----------
    intensity : np.ndarray
        Intensity values in dB per frame
    min_valid_db : float
        Minimum valid intensity value
        
    Returns
    -------
    Dict[str, float]
        Intensity statistics: mean, std, min, max, range
    """
    # Filter out invalid values
    valid_intensity = intensity[intensity > min_valid_db]
    
    if len(valid_intensity) == 0:
        logger.warning("No valid intensity values found")
        return {
            'intensity_mean': 0.0,
            'intensity_std': 0.0,
            'intensity_min': 0.0,
            'intensity_max': 0.0,
            'intensity_range': 0.0
        }
    
    int_mean = float(np.mean(valid_intensity))
    int_std = float(np.std(valid_intensity))
    int_min = float(np.min(valid_intensity))
    int_max = float(np.max(valid_intensity))
    int_range = int_max - int_min
    
    return {
        'intensity_mean': int_mean,
        'intensity_std': int_std,
        'intensity_min': int_min,
        'intensity_max': int_max,
        'intensity_range': int_range
    }


def extract_prosody_features(
    sig: np.ndarray,
    fs: int,
    f0_method: str = 'pyin',
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    use_parselmouth_intensity: bool = False
) -> Dict[str, float]:
    """
    Extract all prosodic features (F0 and intensity).
    
    This is the main interface function for the feature extraction pipeline.
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal (mono, normalized)
    fs : int
        Sample rate
    f0_method : str
        F0 extraction method: 'pyin', 'yin', or 'parselmouth'
    f0_min : float
        Minimum F0 frequency
    f0_max : float
        Maximum F0 frequency
    frame_length : int
        Frame length for spectral analysis
    hop_length : int
        Hop length between frames
    use_parselmouth_intensity : bool
        Whether to use Parselmouth for intensity
        
    Returns
    -------
    Dict[str, float]
        All prosodic features:
        - f0_mean, f0_std, f0_min, f0_max, f0_range
        - intensity_mean, intensity_std, intensity_min, intensity_max, intensity_range
        - voice_rate (voiced segments per second)
    """
    features = {}
    
    # Extract F0
    try:
        f0 = extract_f0(
            sig, fs,
            method=f0_method,
            fmin=f0_min,
            fmax=f0_max,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # Compute F0 statistics
        f0_stats = compute_f0_statistics(f0)
        features.update(f0_stats)
        
        # Compute voice rate
        voice_rate = _compute_voice_rate(f0, hop_length, fs)
        features['voice_rate'] = voice_rate
        
    except Exception as e:
        logger.error(f"Error extracting F0: {e}")
        features.update({
            'f0_mean': 0.0,
            'f0_std': 0.0,
            'f0_min': 0.0,
            'f0_max': 0.0,
            'f0_range': 0.0,
            'voice_rate': 0.0
        })
    
    # Extract intensity
    try:
        intensity = extract_intensity(
            sig, fs,
            frame_length=frame_length,
            hop_length=hop_length,
            use_parselmouth=use_parselmouth_intensity
        )
        
        # Compute intensity statistics
        int_stats = compute_intensity_statistics(intensity)
        features.update(int_stats)
        
    except Exception as e:
        logger.error(f"Error extracting intensity: {e}")
        features.update({
            'intensity_mean': 0.0,
            'intensity_std': 0.0,
            'intensity_min': 0.0,
            'intensity_max': 0.0,
            'intensity_range': 0.0
        })
    
    return features


def _compute_voice_rate(
    f0: np.ndarray,
    hop_length: int,
    fs: int
) -> float:
    """
    Compute voice rate (number of voiced segments per second).
    
    Parameters
    ----------
    f0 : np.ndarray
        F0 values per frame (0 for unvoiced)
    hop_length : int
        Hop length used for F0 extraction
    fs : int
        Sample rate
        
    Returns
    -------
    float
        Voiced segments per second
    """
    # Find voiced frames
    voiced = (f0 > 0).astype(int)
    
    # Find segment boundaries
    diff = np.diff(np.concatenate([[0], voiced, [0]]))
    starts = np.where(diff == 1)[0]
    
    n_voiced_segments = len(starts)
    
    # Calculate total duration
    duration = len(f0) * hop_length / fs
    
    if duration > 0:
        return float(n_voiced_segments / duration)
    return 0.0


def get_prosody_feature_names() -> List[str]:
    """Return list of all prosody feature names."""
    return [
        'f0_mean',
        'f0_std',
        'f0_min',
        'f0_max',
        'f0_range',
        'intensity_mean',
        'intensity_std',
        'intensity_min',
        'intensity_max',
        'intensity_range',
        'voice_rate'
    ]

