"""
Audio loading and preprocessing utilities for Pitt Corpus.

Handles:
- Loading WAV and MP3 files
- Resampling to target sample rate (16 kHz)
- Normalization (DC offset removal, amplitude normalization)
- Optional examiner speech removal using timestamps
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def load_audio(
    audio_path: str,
    target_sr: int = 16000,
    normalize: bool = True,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and optionally resample/normalize.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file (WAV or MP3)
    target_sr : int
        Target sample rate (default: 16000 Hz)
    normalize : bool
        Whether to normalize the audio signal
    mono : bool
        Whether to convert to mono
        
    Returns
    -------
    signal : np.ndarray
        Audio signal as numpy array
    sample_rate : int
        Sample rate of the returned signal
        
    Raises
    ------
    FileNotFoundError
        If audio file doesn't exist
    ValueError
        If audio file is empty or corrupted
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Load audio with librosa (handles both WAV and MP3)
        signal, orig_sr = librosa.load(
            audio_path, 
            sr=None,  # Load at original sample rate first
            mono=mono
        )
        
        # Check if signal is valid
        if signal is None or len(signal) == 0:
            raise ValueError(f"Empty audio signal: {audio_path}")
            
        # Resample if needed
        if orig_sr != target_sr:
            logger.debug(f"Resampling from {orig_sr} Hz to {target_sr} Hz")
            signal = librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)
        
        # Normalize if requested
        if normalize:
            signal = preprocess_audio(signal)
            
        return signal, target_sr
        
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}")
        raise


def preprocess_audio(
    signal: np.ndarray,
    remove_dc: bool = True,
    normalize_amplitude: bool = True,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Preprocess audio signal for feature extraction.
    
    Based on the cross-linguistic paper implementation:
    1. Remove DC offset (subtract mean)
    2. Normalize by maximum absolute value
    
    Parameters
    ----------
    signal : np.ndarray
        Input audio signal
    remove_dc : bool
        Whether to remove DC offset
    normalize_amplitude : bool
        Whether to normalize amplitude to [-1, 1]
    eps : float
        Small value to prevent division by zero
        
    Returns
    -------
    np.ndarray
        Preprocessed signal
    """
    signal = signal.copy()
    
    # Remove DC offset
    if remove_dc:
        signal = signal - np.mean(signal)
    
    # Normalize amplitude
    if normalize_amplitude:
        max_val = np.max(np.abs(signal))
        if max_val > eps:
            signal = signal / max_val
    
    return signal


def convert_mp3_to_wav(
    mp3_path: str,
    wav_path: Optional[str] = None,
    target_sr: int = 16000
) -> str:
    """
    Convert MP3 file to WAV format.
    
    Parameters
    ----------
    mp3_path : str
        Path to input MP3 file
    wav_path : str, optional
        Path for output WAV file. If None, uses same name with .wav extension
    target_sr : int
        Target sample rate for output WAV
        
    Returns
    -------
    str
        Path to output WAV file
    """
    if wav_path is None:
        wav_path = os.path.splitext(mp3_path)[0] + '.wav'
    
    # Load MP3 and save as WAV
    signal, _ = load_audio(mp3_path, target_sr=target_sr, normalize=False)
    sf.write(wav_path, signal, target_sr)
    
    logger.info(f"Converted {mp3_path} to {wav_path}")
    return wav_path


def remove_examiner_speech(
    signal: np.ndarray,
    sample_rate: int,
    examiner_timestamps: List[Tuple[float, float]],
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Remove examiner speech from audio using timestamps.
    
    The examiner (INV) speech segments are either zeroed out or removed,
    keeping only the participant (PAR) speech.
    
    Parameters
    ----------
    signal : np.ndarray
        Audio signal
    sample_rate : int
        Sample rate of the signal
    examiner_timestamps : List[Tuple[float, float]]
        List of (start_time, end_time) tuples for examiner speech (in seconds)
    fill_value : float
        Value to fill examiner regions with (default: 0.0 for silence)
        
    Returns
    -------
    np.ndarray
        Signal with examiner speech removed
    """
    signal = signal.copy()
    
    for start_time, end_time in examiner_timestamps:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure bounds are valid
        start_sample = max(0, start_sample)
        end_sample = min(len(signal), end_sample)
        
        # Zero out examiner speech
        signal[start_sample:end_sample] = fill_value
    
    return signal


def extract_participant_audio(
    signal: np.ndarray,
    sample_rate: int,
    participant_timestamps: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Extract only participant speech segments and concatenate them.
    
    Parameters
    ----------
    signal : np.ndarray
        Audio signal
    sample_rate : int
        Sample rate of the signal
    participant_timestamps : List[Tuple[float, float]]
        List of (start_time, end_time) tuples for participant speech (in seconds)
        
    Returns
    -------
    np.ndarray
        Concatenated participant speech segments
    """
    segments = []
    
    for start_time, end_time in participant_timestamps:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure bounds are valid
        start_sample = max(0, start_sample)
        end_sample = min(len(signal), end_sample)
        
        if end_sample > start_sample:
            segments.append(signal[start_sample:end_sample])
    
    if segments:
        return np.concatenate(segments)
    else:
        logger.warning("No participant segments found, returning original signal")
        return signal


def get_audio_info(audio_path: str) -> dict:
    """
    Get basic information about an audio file.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
        
    Returns
    -------
    dict
        Dictionary with audio information
    """
    try:
        signal, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(signal) / sr
        
        return {
            'path': audio_path,
            'sample_rate': sr,
            'duration_seconds': duration,
            'num_samples': len(signal),
            'max_amplitude': float(np.max(np.abs(signal))),
            'mean_amplitude': float(np.mean(np.abs(signal))),
            'rms': float(np.sqrt(np.mean(signal**2)))
        }
    except Exception as e:
        logger.error(f"Error getting info for {audio_path}: {e}")
        return {'path': audio_path, 'error': str(e)}

