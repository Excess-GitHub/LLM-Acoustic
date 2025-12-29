"""
Voice Quality Features Extraction.

Extracts:
- Jitter: Cycle-to-cycle F0 perturbations
- Shimmer: Cycle-to-cycle amplitude perturbations
- Harmonics-to-Noise Ratio (HNR): Voice quality measure

These features are clinically relevant for detecting:
- Vocal instability (common in aging and neurological conditions)
- Voice quality degradation
- Motor speech impairments
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# Try to import parselmouth (recommended for voice quality)
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logger.warning("Parselmouth not available. Voice quality features may be less accurate.")

import librosa
from scipy import signal as scipy_signal


def compute_jitter(
    sig: np.ndarray,
    fs: int,
    f0: Optional[np.ndarray] = None,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    use_parselmouth: bool = True
) -> Dict[str, float]:
    """
    Compute jitter (F0 perturbation) features.
    
    Jitter measures the cycle-to-cycle variation in fundamental frequency.
    Higher jitter indicates more irregular vocal fold vibration.
    
    Formula (local jitter):
    jitter = mean(|T[i] - T[i-1]|) / mean(T)
    where T[i] is the i-th period (1/F0)
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal
    fs : int
        Sample rate
    f0 : np.ndarray, optional
        Pre-computed F0 contour (if None, will be computed)
    f0_min : float
        Minimum F0 for analysis
    f0_max : float
        Maximum F0 for analysis
    use_parselmouth : bool
        Whether to use Parselmouth (more accurate)
        
    Returns
    -------
    Dict[str, float]
        Jitter features: jitter_mean, jitter_std
    """
    if use_parselmouth and PARSELMOUTH_AVAILABLE:
        return _compute_jitter_parselmouth(sig, fs, f0_min, f0_max)
    else:
        return _compute_jitter_manual(sig, fs, f0, f0_min, f0_max)


def _compute_jitter_parselmouth(
    sig: np.ndarray,
    fs: int,
    f0_min: float,
    f0_max: float
) -> Dict[str, float]:
    """Compute jitter using Parselmouth (Praat)."""
    try:
        sound = parselmouth.Sound(sig, sampling_frequency=fs)
        
        # Create PointProcess for pitch periods
        point_process = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
        
        # Get jitter (local) - relative average perturbation
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Get jitter (local, absolute) for std computation
        jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Handle NaN
        if np.isnan(jitter_local):
            jitter_local = 0.0
        if np.isnan(jitter_abs):
            jitter_abs = 0.0
            
        return {
            'jitter_mean': float(jitter_local),
            'jitter_std': float(jitter_abs)  # Using absolute jitter as std proxy
        }
        
    except Exception as e:
        logger.warning(f"Parselmouth jitter computation failed: {e}")
        return {'jitter_mean': 0.0, 'jitter_std': 0.0}


def _compute_jitter_manual(
    sig: np.ndarray,
    fs: int,
    f0: Optional[np.ndarray],
    f0_min: float,
    f0_max: float
) -> Dict[str, float]:
    """Compute jitter manually from F0 contour."""
    # Extract F0 if not provided
    if f0 is None:
        try:
            f0, _, _ = librosa.pyin(
                sig, fmin=f0_min, fmax=f0_max, sr=fs
            )
            f0 = np.nan_to_num(f0, nan=0.0)
        except Exception as e:
            logger.warning(f"F0 extraction failed: {e}")
            return {'jitter_mean': 0.0, 'jitter_std': 0.0}
    
    # Get voiced F0 values
    voiced_f0 = f0[f0 > 0]
    
    if len(voiced_f0) < 2:
        return {'jitter_mean': 0.0, 'jitter_std': 0.0}
    
    # Convert F0 to periods
    periods = 1.0 / voiced_f0
    
    # Compute period differences
    period_diffs = np.abs(np.diff(periods))
    
    # Compute jitter
    mean_period = np.mean(periods)
    if mean_period > 0:
        jitter_values = period_diffs / mean_period
        jitter_mean = float(np.mean(jitter_values))
        jitter_std = float(np.std(jitter_values))
    else:
        jitter_mean = 0.0
        jitter_std = 0.0
    
    return {
        'jitter_mean': jitter_mean,
        'jitter_std': jitter_std
    }


def compute_shimmer(
    sig: np.ndarray,
    fs: int,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    use_parselmouth: bool = True
) -> Dict[str, float]:
    """
    Compute shimmer (amplitude perturbation) features.
    
    Shimmer measures the cycle-to-cycle variation in amplitude.
    Higher shimmer indicates more irregular vocal fold vibration amplitude.
    
    Formula (local shimmer):
    shimmer = mean(|A[i] - A[i-1]|) / mean(A)
    where A[i] is the amplitude of the i-th period
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal
    fs : int
        Sample rate
    f0_min : float
        Minimum F0 for pitch period detection
    f0_max : float
        Maximum F0 for pitch period detection
    use_parselmouth : bool
        Whether to use Parselmouth (more accurate)
        
    Returns
    -------
    Dict[str, float]
        Shimmer features: shimmer_mean, shimmer_std
    """
    if use_parselmouth and PARSELMOUTH_AVAILABLE:
        return _compute_shimmer_parselmouth(sig, fs, f0_min, f0_max)
    else:
        return _compute_shimmer_manual(sig, fs, f0_min, f0_max)


def _compute_shimmer_parselmouth(
    sig: np.ndarray,
    fs: int,
    f0_min: float,
    f0_max: float
) -> Dict[str, float]:
    """Compute shimmer using Parselmouth (Praat)."""
    try:
        sound = parselmouth.Sound(sig, sampling_frequency=fs)
        
        # Create PointProcess for pitch periods
        point_process = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
        
        # Get shimmer (local) - relative amplitude perturbation
        shimmer_local = call(
            [sound, point_process], "Get shimmer (local)", 
            0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        
        # Get shimmer (apq3) as another measure
        shimmer_apq3 = call(
            [sound, point_process], "Get shimmer (apq3)", 
            0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        
        # Handle NaN
        if np.isnan(shimmer_local):
            shimmer_local = 0.0
        if np.isnan(shimmer_apq3):
            shimmer_apq3 = 0.0
            
        return {
            'shimmer_mean': float(shimmer_local),
            'shimmer_std': float(shimmer_apq3)  # Using APQ3 as variability measure
        }
        
    except Exception as e:
        logger.warning(f"Parselmouth shimmer computation failed: {e}")
        return {'shimmer_mean': 0.0, 'shimmer_std': 0.0}


def _compute_shimmer_manual(
    sig: np.ndarray,
    fs: int,
    f0_min: float,
    f0_max: float
) -> Dict[str, float]:
    """Compute shimmer manually using peak amplitudes."""
    try:
        # Get F0 to find pitch periods
        f0, voiced_flag, _ = librosa.pyin(
            sig, fmin=f0_min, fmax=f0_max, sr=fs
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Find voiced segments
        voiced_indices = np.where(f0 > 0)[0]
        if len(voiced_indices) < 2:
            return {'shimmer_mean': 0.0, 'shimmer_std': 0.0}
        
        # Compute amplitude envelope
        envelope = np.abs(scipy_signal.hilbert(sig))
        
        # Sample envelope at frame centers (using hop length of 512)
        hop_length = 512
        frame_centers = voiced_indices * hop_length
        frame_centers = frame_centers[frame_centers < len(envelope)]
        
        if len(frame_centers) < 2:
            return {'shimmer_mean': 0.0, 'shimmer_std': 0.0}
        
        amplitudes = envelope[frame_centers]
        
        # Compute amplitude differences
        amp_diffs = np.abs(np.diff(amplitudes))
        
        mean_amp = np.mean(amplitudes)
        if mean_amp > 0:
            shimmer_values = amp_diffs / mean_amp
            shimmer_mean = float(np.mean(shimmer_values))
            shimmer_std = float(np.std(shimmer_values))
        else:
            shimmer_mean = 0.0
            shimmer_std = 0.0
        
        return {
            'shimmer_mean': shimmer_mean,
            'shimmer_std': shimmer_std
        }
        
    except Exception as e:
        logger.warning(f"Manual shimmer computation failed: {e}")
        return {'shimmer_mean': 0.0, 'shimmer_std': 0.0}


def compute_hnr(
    sig: np.ndarray,
    fs: int,
    time_step: float = 0.01,
    min_pitch: float = 75.0,
    silence_threshold: float = 0.1,
    use_parselmouth: bool = True
) -> Dict[str, float]:
    """
    Compute Harmonics-to-Noise Ratio (HNR) features.
    
    HNR measures the ratio of harmonic (periodic) energy to noise energy.
    Higher HNR indicates a clearer, more harmonic voice.
    
    Formula:
    HNR[dB] = 10 * log10(harmonic_energy / noise_energy)
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal
    fs : int
        Sample rate
    time_step : float
        Time step for HNR computation (seconds)
    min_pitch : float
        Minimum pitch for analysis
    silence_threshold : float
        Silence threshold
    use_parselmouth : bool
        Whether to use Parselmouth
        
    Returns
    -------
    Dict[str, float]
        HNR features: hnr_mean, hnr_std, hnr_min, hnr_max
    """
    if use_parselmouth and PARSELMOUTH_AVAILABLE:
        return _compute_hnr_parselmouth(sig, fs, time_step, min_pitch, silence_threshold)
    else:
        return _compute_hnr_manual(sig, fs)


def _compute_hnr_parselmouth(
    sig: np.ndarray,
    fs: int,
    time_step: float,
    min_pitch: float,
    silence_threshold: float
) -> Dict[str, float]:
    """Compute HNR using Parselmouth (Praat)."""
    try:
        sound = parselmouth.Sound(sig, sampling_frequency=fs)
        
        # Compute harmonicity
        harmonicity = call(
            sound, "To Harmonicity (cc)", 
            time_step, min_pitch, silence_threshold, 1.0
        )
        
        # Get HNR values
        n_frames = call(harmonicity, "Get number of frames")
        hnr_values = []
        
        for i in range(1, n_frames + 1):
            hnr = call(harmonicity, "Get value in frame", i)
            if not np.isnan(hnr) and hnr > -200:  # Filter out undefined values
                hnr_values.append(hnr)
        
        if len(hnr_values) == 0:
            return {
                'hnr_mean': 0.0,
                'hnr_std': 0.0,
                'hnr_min': 0.0,
                'hnr_max': 0.0
            }
        
        hnr_array = np.array(hnr_values)
        
        return {
            'hnr_mean': float(np.mean(hnr_array)),
            'hnr_std': float(np.std(hnr_array)),
            'hnr_min': float(np.min(hnr_array)),
            'hnr_max': float(np.max(hnr_array))
        }
        
    except Exception as e:
        logger.warning(f"Parselmouth HNR computation failed: {e}")
        return {
            'hnr_mean': 0.0,
            'hnr_std': 0.0,
            'hnr_min': 0.0,
            'hnr_max': 0.0
        }


def _compute_hnr_manual(
    sig: np.ndarray,
    fs: int,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Dict[str, float]:
    """
    Compute HNR manually using autocorrelation method.
    
    This is a simplified version that estimates HNR from
    the ratio of harmonic to total energy.
    """
    try:
        # Compute spectral features
        stft = librosa.stft(sig, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Estimate harmonic content using harmonic-percussive separation
        harmonic, percussive = librosa.decompose.hpss(stft)
        harmonic_mag = np.abs(harmonic)
        
        # Compute energy
        total_energy = np.sum(magnitude ** 2, axis=0)
        harmonic_energy = np.sum(harmonic_mag ** 2, axis=0)
        
        # Noise energy
        noise_energy = total_energy - harmonic_energy
        noise_energy = np.maximum(noise_energy, 1e-10)  # Avoid division by zero
        
        # HNR in dB
        hnr = 10 * np.log10(harmonic_energy / noise_energy)
        
        # Filter valid values
        valid_hnr = hnr[np.isfinite(hnr)]
        
        if len(valid_hnr) == 0:
            return {
                'hnr_mean': 0.0,
                'hnr_std': 0.0,
                'hnr_min': 0.0,
                'hnr_max': 0.0
            }
        
        return {
            'hnr_mean': float(np.mean(valid_hnr)),
            'hnr_std': float(np.std(valid_hnr)),
            'hnr_min': float(np.min(valid_hnr)),
            'hnr_max': float(np.max(valid_hnr))
        }
        
    except Exception as e:
        logger.warning(f"Manual HNR computation failed: {e}")
        return {
            'hnr_mean': 0.0,
            'hnr_std': 0.0,
            'hnr_min': 0.0,
            'hnr_max': 0.0
        }


def extract_voice_quality_features(
    sig: np.ndarray,
    fs: int,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    use_parselmouth: bool = True,
    hnr_time_step: float = 0.01,
    hnr_min_pitch: float = 75.0,
    hnr_silence_threshold: float = 0.1
) -> Dict[str, float]:
    """
    Extract all voice quality features.
    
    This is the main interface function for the feature extraction pipeline.
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal (mono, normalized)
    fs : int
        Sample rate
    f0_min : float
        Minimum F0 for analysis
    f0_max : float
        Maximum F0 for analysis
    use_parselmouth : bool
        Whether to use Parselmouth (recommended for accuracy)
    hnr_time_step : float
        Time step for HNR computation
    hnr_min_pitch : float
        Minimum pitch for HNR
    hnr_silence_threshold : float
        Silence threshold for HNR
        
    Returns
    -------
    Dict[str, float]
        All voice quality features:
        - jitter_mean, jitter_std
        - shimmer_mean, shimmer_std
        - hnr_mean, hnr_std, hnr_min, hnr_max
    """
    features = {}
    
    # Compute jitter
    try:
        jitter_feats = compute_jitter(
            sig, fs, 
            f0_min=f0_min, 
            f0_max=f0_max,
            use_parselmouth=use_parselmouth
        )
        features.update(jitter_feats)
    except Exception as e:
        logger.error(f"Error computing jitter: {e}")
        features.update({'jitter_mean': 0.0, 'jitter_std': 0.0})
    
    # Compute shimmer
    try:
        shimmer_feats = compute_shimmer(
            sig, fs,
            f0_min=f0_min,
            f0_max=f0_max,
            use_parselmouth=use_parselmouth
        )
        features.update(shimmer_feats)
    except Exception as e:
        logger.error(f"Error computing shimmer: {e}")
        features.update({'shimmer_mean': 0.0, 'shimmer_std': 0.0})
    
    # Compute HNR
    try:
        hnr_feats = compute_hnr(
            sig, fs,
            time_step=hnr_time_step,
            min_pitch=hnr_min_pitch,
            silence_threshold=hnr_silence_threshold,
            use_parselmouth=use_parselmouth
        )
        features.update(hnr_feats)
    except Exception as e:
        logger.error(f"Error computing HNR: {e}")
        features.update({
            'hnr_mean': 0.0,
            'hnr_std': 0.0,
            'hnr_min': 0.0,
            'hnr_max': 0.0
        })
    
    return features


def get_voice_quality_feature_names() -> List[str]:
    """Return list of all voice quality feature names."""
    return [
        'jitter_mean',
        'jitter_std',
        'shimmer_mean',
        'shimmer_std',
        'hnr_mean',
        'hnr_std',
        'hnr_min',
        'hnr_max'
    ]

