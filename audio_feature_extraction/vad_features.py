"""
Voice Activity Detection (VAD) and timing features.

Based on the cross-linguistic study's Timing_VAD.py implementation.
Implements energy-based VAD with pause/speech segmentation and
statistical features extraction.

Reference: Timing_VAD.py by Tomas & Paula (2021)
"""

import numpy as np
import scipy as sp
from scipy import signal
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def extract_windows(signal_array: np.ndarray, size: int, step: int) -> np.ndarray:
    """
    Extract overlapping windows from a 1D signal.
    
    Directly adapted from Timing_VAD.py extract_windows() function.
    
    Parameters
    ----------
    signal_array : np.ndarray
        Input 1D signal
    size : int
        Window size in samples
    step : int
        Step size (hop) in samples
        
    Returns
    -------
    np.ndarray
        2D array where each row is a window
    """
    # Ensure mono signal
    assert signal_array.ndim == 1, "Signal must be 1D"
    
    n_frames = int((len(signal_array) - size) / step)
    
    # Extract frames
    windows = [signal_array[i * step : i * step + size] 
               for i in range(n_frames)]
    
    if len(windows) == 0:
        return np.array([]).reshape(0, size)
    
    return np.vstack(windows)


def get_segments(
    sig: np.ndarray,
    fs: int,
    segments: np.ndarray,
    is_pause: bool = False
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Extract segment durations, audio segments, and timestamps from binary labels.
    
    Directly adapted from Timing_VAD.py get_segments() function.
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal
    fs : int
        Sample rate
    segments : np.ndarray
        Binary labels (same length as sig): 1 for segment, 0 otherwise
    is_pause : bool
        Whether these are pause segments (for potential filtering)
        
    Returns
    -------
    seg_dur : np.ndarray
        Array of segment durations in seconds
    seg_list : List[np.ndarray]
        List of audio segments
    seg_time : np.ndarray
        Array of [start, end] timestamps in seconds
    """
    # Ensure boundaries are 0
    segments = segments.copy()
    segments[0] = 0
    segments[-1] = 0
    
    # Find segment boundaries using diff
    ydf = np.diff(segments)
    lim_end = np.where(ydf == -1)[0] + 1
    lim_ini = np.where(ydf == 1)[0] + 1
    
    # Handle edge case where no segments found
    if len(lim_ini) == 0 or len(lim_end) == 0:
        return np.array([0.0]), [], np.array([[0.0, 0.0]])
    
    # Ensure same number of starts and ends
    min_len = min(len(lim_ini), len(lim_end))
    lim_ini = lim_ini[:min_len]
    lim_end = lim_end[:min_len]
    
    # Extract segments
    seg_dur = []
    seg_list = []
    seg_time = []
    
    for idx in range(len(lim_ini)):
        tini = lim_ini[idx] / fs
        tend = lim_end[idx] / fs
        seg_dur.append(abs(tend - tini))
        seg_list.append(sig[lim_ini[idx]:lim_end[idx]])
        seg_time.append([lim_ini[idx], lim_end[idx]])
    
    seg_dur = np.asarray(seg_dur)
    seg_time = np.vstack(seg_time) / fs if seg_time else np.array([[0.0, 0.0]])
    
    return seg_dur, seg_list, seg_time


def eVAD(
    sig: np.ndarray,
    fs: int,
    win: float = 0.025,
    step: float = 0.01
) -> Dict[str, Any]:
    """
    Energy-based Voice Activity Detection.
    
    Directly adapted from Timing_VAD.py eVAD() function.
    
    Implementation steps:
    1. Normalize signal (remove DC, scale to [-1, 1])
    2. Add silence padding at start/end for robust boundary detection
    3. Extract frames with Hanning window
    4. Compute log-energy per frame
    5. Apply Gaussian smoothing to energy contour
    6. Set threshold as median of negative energy values
    7. Generate binary speech/pause labels
    8. Extract segment boundaries and durations
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal (mono)
    fs : int
        Sample rate in Hz
    win : float
        Window size in seconds (default: 0.025 = 25ms)
    step : float
        Hop size in seconds (default: 0.01 = 10ms)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - Pause_labels: Binary array (1=pause, 0=speech)
        - Pause_duration: Array of pause durations (seconds)
        - Pause_segments: List of pause audio segments
        - Pause_times: Array of [start, end] for each pause
        - Speech_labels: Binary array (1=speech, 0=pause)
        - Speech_duration: Array of speech durations (seconds)
        - Speech_segments: List of speech audio segments
        - Speech_times: Array of [start, end] for each speech segment
    """
    # Normalize signal
    sig = sig - np.mean(sig)
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = sig / max_val
    
    lsig = len(sig)
    
    # Find minimum threshold based on quietest part of signal
    e = []
    frames = extract_windows(sig, int(win * fs), int(step * fs))
    
    if len(frames) == 0:
        logger.warning("No frames extracted from signal")
        return _empty_vad_result(lsig)
    
    for seg in frames:
        energy = np.sum(np.absolute(seg) ** 2) / len(seg)
        if energy > 0:
            e.append(10 * np.log10(energy))
        else:
            e.append(-100)  # Very low energy for silence
    
    e = np.asarray(e)
    idx_min = np.where(e == np.min(e))[0]
    if len(idx_min) > 0 and idx_min[0] < len(frames):
        thr = np.min(frames[idx_min[0]])
    else:
        thr = 1e-10
    
    # Add silence padding at start and end
    ext_sil = int(fs)  # 1 second of padding
    esil = int((ext_sil / 2) / fs / step)
    
    # Create padded signal with random noise at threshold level
    new_sig = np.random.randn(lsig + ext_sil) * max(abs(thr), 1e-10)
    new_sig[int(ext_sil / 2):lsig + int(ext_sil / 2)] = sig
    sig_padded = new_sig
    
    # Recompute energy on padded signal with Hanning window
    e = []
    frames = extract_windows(sig_padded, int(win * fs), int(step * fs))
    
    if len(frames) == 0:
        return _empty_vad_result(lsig)
    
    frames = frames * np.hanning(int(win * fs))
    
    for seg in frames:
        energy = np.sum(np.absolute(seg) ** 2) / len(seg)
        if energy > 0:
            e.append(10 * np.log10(energy))
        else:
            e.append(-100)
    
    e = np.asarray(e)
    e = e - np.mean(e)  # Remove DC offset from energy
    
    # Gaussian smoothing of energy contour
    gauslen = int(fs * 0.01)  # 10ms kernel
    if gauslen < 3:
        gauslen = 3
    window = signal.windows.gaussian(gauslen, std=int(gauslen * 0.05) or 1)
    
    # Convolve with Gaussian for smoothing
    smooth_env = np.convolve(e, window)
    max_smooth = np.max(smooth_env)
    if max_smooth > 0:
        smooth_env = smooth_env / max_smooth
    
    # Trim convolution edges
    ini = int(gauslen / 2)
    fin = len(smooth_env) - ini
    e = smooth_env[ini:fin]
    
    # Normalize
    max_e = np.max(np.abs(e))
    if max_e > 0:
        e = e / max_e
    
    # Trim to original length (removing padding effects)
    e = e[esil:int(lsig / fs / step) + esil]
    
    # Set threshold as median of negative energy values
    neg_e = e[e < 0]
    if len(neg_e) > 0:
        thr = np.median(neg_e)
    else:
        thr = 0
    
    # Generate binary labels
    cont_sil = np.zeros(lsig)  # Silence/pause labels
    cont_vad = np.zeros(lsig)  # Speech labels
    
    itime = 0
    etime = int(win * fs)
    
    for i in range(len(e)):
        if e[i] <= thr:
            cont_sil[itime:etime] = 1
        else:
            cont_vad[itime:etime] = 1
        
        itime = i * int(step * fs)
        etime = itime + int(win * fs)
    
    # Extract segments
    if np.sum(cont_sil) != 0:
        # Pauses
        dur_sil, seg_sil, time_sil = get_segments(sig, fs, cont_sil, True)
        # Speech
        dur_vad, seg_vad, time_vad = get_segments(sig, fs, cont_vad)
    else:
        dur_sil = np.array([0.0])
        seg_sil = []
        time_sil = np.array([[0.0, 0.0]])
        dur_vad = np.array([0.0])
        seg_vad = []
        time_vad = np.array([[0.0, 0.0]])
    
    return {
        'Pause_labels': cont_sil,
        'Pause_duration': dur_sil,
        'Pause_segments': seg_sil,
        'Pause_times': time_sil,
        'Speech_labels': cont_vad,
        'Speech_duration': dur_vad,
        'Speech_segments': seg_vad,
        'Speech_times': time_vad
    }


def _empty_vad_result(lsig: int) -> Dict[str, Any]:
    """Return empty VAD result for edge cases."""
    return {
        'Pause_labels': np.zeros(lsig),
        'Pause_duration': np.array([0.0]),
        'Pause_segments': [],
        'Pause_times': np.array([[0.0, 0.0]]),
        'Speech_labels': np.zeros(lsig),
        'Speech_duration': np.array([0.0]),
        'Speech_segments': [],
        'Speech_times': np.array([[0.0, 0.0]])
    }


def duration_feats(
    x: Dict[str, Any],
    sig: np.ndarray,
    fs: int
) -> np.ndarray:
    """
    Extract duration features from VAD output.
    
    Directly adapted from Timing_VAD.py duration_feats() function.
    
    Features extracted:
    1. pause_len_ratio: Number of pauses / recording duration
    2. speech_len_ratio: Number of speech segments / recording duration
    3. pause_speech_ratio: Number of speech segments / number of pauses
    4. Speech duration statistics: mean, std, skew, kurtosis, min, max
    5. Pause duration statistics: mean, std, skew, kurtosis, min, max
    
    Parameters
    ----------
    x : Dict[str, Any]
        VAD output dictionary from eVAD()
    sig : np.ndarray
        Audio signal
    fs : int
        Sample rate
        
    Returns
    -------
    np.ndarray
        Array of 15 features: [pause_len, speech_len, sp_ps, 
                              speech_mean/std/skew/kurt/min/max,
                              pause_mean/std/skew/kurt/min/max]
    """
    recording_duration = len(sig) / fs
    
    # Handle edge cases
    n_pauses = len(x['Pause_duration'])
    n_speech = len(x['Speech_duration'])
    
    # Ratios
    pause_len = n_pauses / recording_duration if recording_duration > 0 else 0
    speech_len = n_speech / recording_duration if recording_duration > 0 else 0
    sp_ps = n_speech / n_pauses if n_pauses > 0 else n_speech
    
    # Speech duration statistics
    speech_dur = x['Speech_duration']
    if len(speech_dur) > 0 and np.sum(speech_dur) > 0:
        speech_stats = np.array([
            np.mean(speech_dur),
            np.std(speech_dur),
            stats.skew(speech_dur) if len(speech_dur) > 2 else 0,
            stats.kurtosis(speech_dur) if len(speech_dur) > 3 else 0,
            np.min(speech_dur),
            np.max(speech_dur)
        ])
    else:
        speech_stats = np.zeros(6)
    
    # Pause duration statistics
    pause_dur = x['Pause_duration']
    if len(pause_dur) > 0 and np.sum(pause_dur) > 0:
        pause_stats = np.array([
            np.mean(pause_dur),
            np.std(pause_dur),
            stats.skew(pause_dur) if len(pause_dur) > 2 else 0,
            stats.kurtosis(pause_dur) if len(pause_dur) > 3 else 0,
            np.min(pause_dur),
            np.max(pause_dur)
        ])
    else:
        pause_stats = np.zeros(6)
    
    return np.hstack((pause_len, speech_len, sp_ps, speech_stats, pause_stats))


def extract_vad_features(
    sig: np.ndarray,
    fs: int,
    win: float = 0.025,
    step: float = 0.01
) -> Dict[str, float]:
    """
    Extract all VAD-based timing features as a dictionary.
    
    This is the main interface function for the feature extraction pipeline.
    
    Parameters
    ----------
    sig : np.ndarray
        Audio signal (mono, normalized)
    fs : int
        Sample rate in Hz
    win : float
        VAD window size in seconds
    step : float
        VAD hop size in seconds
        
    Returns
    -------
    Dict[str, float]
        Dictionary of feature names -> values:
        - pause_ratio: Total pause time / total recording duration
        - speech_ratio: Total speech time / total recording duration
        - pause_speech_ratio: Number of speech segments / number of pauses
        - num_pauses_per_sec: Total pause count / recording duration
        - num_speech_segments_per_sec: Total speech segment count / recording duration
        - pause_dur_mean, pause_dur_std, pause_dur_skew, pause_dur_kurt, pause_dur_min, pause_dur_max
        - speech_dur_mean, speech_dur_std, speech_dur_skew, speech_dur_kurt, speech_dur_min, speech_dur_max
    """
    # Run VAD
    vad_result = eVAD(sig, fs, win=win, step=step)
    
    # Get duration features from original function
    raw_feats = duration_feats(vad_result, sig, fs)
    
    # Calculate additional features
    recording_duration = len(sig) / fs
    
    # Total pause and speech time
    total_pause_time = np.sum(vad_result['Pause_duration'])
    total_speech_time = np.sum(vad_result['Speech_duration'])
    total_time = total_pause_time + total_speech_time
    
    # Ratios based on time (not counts)
    pause_ratio = total_pause_time / recording_duration if recording_duration > 0 else 0
    speech_ratio = total_speech_time / recording_duration if recording_duration > 0 else 0
    
    # Counts per second (from raw_feats)
    num_pauses_per_sec = raw_feats[0]  # pause_len from duration_feats
    num_speech_segments_per_sec = raw_feats[1]  # speech_len from duration_feats
    pause_speech_ratio = raw_feats[2]  # sp_ps from duration_feats
    
    # Build feature dictionary
    features = {
        # Whole-recording features
        'pause_ratio': float(pause_ratio),
        'speech_ratio': float(speech_ratio),
        'pause_speech_ratio': float(pause_speech_ratio),
        'num_pauses_per_sec': float(num_pauses_per_sec),
        'num_speech_segments_per_sec': float(num_speech_segments_per_sec),
        
        # Speech duration statistics (indices 3-8 from raw_feats)
        'speech_dur_mean': float(raw_feats[3]),
        'speech_dur_std': float(raw_feats[4]),
        'speech_dur_skew': float(raw_feats[5]),
        'speech_dur_kurt': float(raw_feats[6]),
        'speech_dur_min': float(raw_feats[7]),
        'speech_dur_max': float(raw_feats[8]),
        
        # Pause duration statistics (indices 9-14 from raw_feats)
        'pause_dur_mean': float(raw_feats[9]),
        'pause_dur_std': float(raw_feats[10]),
        'pause_dur_skew': float(raw_feats[11]),
        'pause_dur_kurt': float(raw_feats[12]),
        'pause_dur_min': float(raw_feats[13]),
        'pause_dur_max': float(raw_feats[14]),
    }
    
    # Handle NaN values
    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            features[key] = 0.0
            logger.debug(f"Replaced NaN/Inf in {key}")
    
    return features


def compute_voice_rate(
    f0_contour: np.ndarray,
    hop_length: int,
    fs: int
) -> float:
    """
    Compute voice rate (number of voiced segments per second).
    
    Parameters
    ----------
    f0_contour : np.ndarray
        F0 values per frame (0 for unvoiced)
    hop_length : int
        Hop length used for F0 extraction
    fs : int
        Sample rate
        
    Returns
    -------
    float
        Number of voiced segments per second
    """
    # Find voiced frames
    voiced = (f0_contour > 0).astype(int)
    
    # Find segment boundaries
    diff = np.diff(np.concatenate([[0], voiced, [0]]))
    starts = np.where(diff == 1)[0]
    
    n_voiced_segments = len(starts)
    
    # Calculate duration
    duration = len(f0_contour) * hop_length / fs
    
    if duration > 0:
        return n_voiced_segments / duration
    return 0.0


def get_vad_feature_names() -> List[str]:
    """Return list of all VAD feature names in order."""
    return [
        'pause_ratio',
        'speech_ratio', 
        'pause_speech_ratio',
        'num_pauses_per_sec',
        'num_speech_segments_per_sec',
        'speech_dur_mean',
        'speech_dur_std',
        'speech_dur_skew',
        'speech_dur_kurt',
        'speech_dur_min',
        'speech_dur_max',
        'pause_dur_mean',
        'pause_dur_std',
        'pause_dur_skew',
        'pause_dur_kurt',
        'pause_dur_min',
        'pause_dur_max',
    ]

