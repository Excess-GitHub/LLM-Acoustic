"""
VGGish audio feature extraction.

VGGish is a variant of VGG for audio processing that generates 128-dimensional
embeddings from audio windows (0.96 seconds each).

Based on: "VGGish: A VGG-like Audio Classification Model" (Hershey et al., 2017)
"""

import os
from typing import Optional, List, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm


class VGGishFeatureExtractor:
    """
    Extract VGGish features from audio files.
    
    VGGish generates 128-dimensional embeddings from 0.96s audio windows.
    Uses pre-trained weights from AudioSet.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[str] = None,
        use_fallback: bool = False
    ):
        """
        Initialize VGGish feature extractor.
        
        Args:
            sample_rate: Target sample rate (VGGish expects 16kHz).
            device: Device to run on ('cuda', 'cpu', or None for auto).
            use_fallback: Force use of mel-spectrogram fallback (lighter on memory).
        """
        self.sample_rate = sample_rate
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.use_torch_vggish = False
        
        if use_fallback:
            # Use lightweight fallback when explicitly requested
            print(f"Using mel-spectrogram features (fallback mode)")
            self._setup_fallback()
        else:
            self._load_model()
    
    def _load_model(self):
        """Load VGGish model from torch hub."""
        try:
            # Try loading from torch hub
            print("Loading VGGish model...")
            self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
            self.model.eval()
            self.model = self.model.to(self.device)
            print(f"VGGish model loaded on {self.device}")
            self.use_torch_vggish = True
        except Exception as e:
            print(f"Warning: Could not load VGGish from torch hub: {e}")
            print("Using fallback mel-spectrogram feature extraction")
            self.use_torch_vggish = False
            self._setup_fallback()
    
    def _load_model_safe(self):
        """Load VGGish model with better error handling for low memory."""
        # Use fallback directly for CPU to avoid potential issues
        if self.device == 'cpu':
            print("Using mel-spectrogram features (CPU mode)")
            self.use_torch_vggish = False
            self._setup_fallback()
            return
        
        self._load_model()
    
    def _setup_fallback(self):
        """Setup fallback mel-spectrogram extractor."""
        # VGGish-like parameters
        self.n_fft = 400  # 25ms at 16kHz
        self.hop_length = 160  # 10ms
        self.n_mels = 64
        self.window_samples = int(0.96 * self.sample_rate)  # 0.96s
        self.hop_samples = int(0.48 * self.sample_rate)  # 0.48s (50% overlap)
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        ).to(self.device)
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)
    
    def _extract_vggish_embeddings(
        self,
        waveform: torch.Tensor
    ) -> np.ndarray:
        """
        Extract VGGish embeddings using pretrained model.
        
        Returns:
            Array of shape (n_windows, 128) containing embeddings.
        """
        with torch.no_grad():
            # VGGish expects numpy array
            audio_np = waveform.cpu().numpy()
            embeddings = self.model.forward(audio_np, self.sample_rate)
        
        return embeddings.cpu().numpy()
    
    def _extract_mel_embeddings(
        self,
        waveform: torch.Tensor
    ) -> np.ndarray:
        """
        Extract mel-spectrogram based embeddings (fallback).
        
        Returns:
            Array of shape (n_windows, 128) containing embeddings.
        """
        waveform = waveform.to(self.device)
        
        # Split into windows
        n_samples = waveform.shape[0]
        embeddings = []
        
        for start in range(0, n_samples - self.window_samples + 1, self.hop_samples):
            end = start + self.window_samples
            window = waveform[start:end]
            
            # Compute mel spectrogram
            mel_spec = self.mel_transform(window.unsqueeze(0))
            mel_spec = torch.log(mel_spec + 1e-9)
            
            # Average over time to get 64-dim, then project to 128-dim
            mel_avg = mel_spec.mean(dim=-1).squeeze()  # (n_mels,)
            
            # Simple projection to 128 dimensions
            embedding = torch.cat([mel_avg, mel_avg])  # (128,)
            embeddings.append(embedding.cpu().numpy())
        
        if len(embeddings) == 0:
            # Handle very short audio
            mel_spec = self.mel_transform(waveform.unsqueeze(0))
            mel_spec = torch.log(mel_spec + 1e-9)
            mel_avg = mel_spec.mean(dim=-1).squeeze()
            embedding = torch.cat([mel_avg, mel_avg])
            embeddings.append(embedding.cpu().numpy())
        
        return np.stack(embeddings)
    
    def extract(
        self,
        audio_path: str
    ) -> np.ndarray:
        """
        Extract VGGish features from audio file.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Array of shape (n_windows, 128) containing VGGish embeddings.
        """
        waveform = self._load_audio(audio_path)
        
        if self.use_torch_vggish:
            return self._extract_vggish_embeddings(waveform)
        else:
            return self._extract_mel_embeddings(waveform)
    
    def extract_batch(
        self,
        audio_paths: List[str],
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Extract VGGish features from multiple audio files.
        
        Args:
            audio_paths: List of audio file paths.
            show_progress: Show progress bar.
            
        Returns:
            List of embedding arrays.
        """
        results = []
        iterator = tqdm(audio_paths, desc="Extracting VGGish") if show_progress else audio_paths
        
        for audio_path in iterator:
            try:
                embeddings = self.extract(audio_path)
                results.append(embeddings)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Return empty embeddings
                results.append(np.zeros((1, 128)))
        
        return results


class MelSpectrogramExtractor:
    """
    Alternative audio feature extractor using mel spectrograms.
    
    Can be used as a lighter alternative to VGGish.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        window_seconds: float = 0.96,
        hop_seconds: float = 0.48,
        device: Optional[str] = None
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_samples = int(window_seconds * sample_rate)
        self.hop_samples = int(hop_seconds * sample_rate)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        ).to(self.device)
    
    def extract(self, audio_path: str) -> np.ndarray:
        """Extract mel spectrogram features."""
        waveform, sr = torchaudio.load(audio_path)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        waveform = waveform.to(self.device).squeeze(0)
        n_samples = waveform.shape[0]
        
        embeddings = []
        for start in range(0, n_samples - self.window_samples + 1, self.hop_samples):
            end = start + self.window_samples
            window = waveform[start:end]
            
            mel_spec = self.mel_transform(window.unsqueeze(0))
            mel_spec = torch.log(mel_spec + 1e-9)
            
            # Average over time
            embedding = mel_spec.mean(dim=-1).squeeze()
            embeddings.append(embedding.cpu().numpy())
        
        if len(embeddings) == 0:
            mel_spec = self.mel_transform(waveform.unsqueeze(0))
            mel_spec = torch.log(mel_spec + 1e-9)
            embedding = mel_spec.mean(dim=-1).squeeze()
            embeddings.append(embedding.cpu().numpy())
        
        return np.stack(embeddings)


if __name__ == "__main__":
    # Test VGGish extraction
    extractor = VGGishFeatureExtractor()
    
    test_audio = "../Pitt Corpus/Pitt Corpus/Media/Dementia/Cookie/WAV/001-0.wav"
    if os.path.exists(test_audio):
        embeddings = extractor.extract(test_audio)
        print(f"VGGish embeddings shape: {embeddings.shape}")
        print(f"  Number of windows: {embeddings.shape[0]}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")

