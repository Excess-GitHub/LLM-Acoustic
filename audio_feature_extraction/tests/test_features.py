"""
Unit tests for audio feature extraction pipeline.

Tests cover:
- VAD feature extraction
- Prosody feature extraction
- Voice quality feature extraction
- Audio loading and preprocessing
- Metadata parsing

Run with: pytest tests/test_features.py -v
"""

import pytest
import numpy as np
import os
import tempfile
import soundfile as sf
from pathlib import Path

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vad_features import (
    extract_windows,
    eVAD,
    duration_feats,
    extract_vad_features,
    get_vad_feature_names
)
from prosody_features import (
    extract_f0,
    extract_intensity,
    extract_prosody_features,
    compute_f0_statistics,
    get_prosody_feature_names
)
from voice_quality_features import (
    compute_jitter,
    compute_shimmer,
    compute_hnr,
    extract_voice_quality_features,
    get_voice_quality_feature_names
)
from utils.audio_loader import load_audio, preprocess_audio
from utils.validation import validate_features, FEATURE_RANGES


# ============== Fixtures ==============

@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 16000


@pytest.fixture
def test_signal(sample_rate):
    """Generate a test signal with speech-like characteristics."""
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a signal with voiced segments (sine waves) and pauses (near silence)
    signal = np.zeros_like(t)
    
    # Add voiced segments with fundamental frequency around 150 Hz
    f0 = 150  # Hz
    
    # Voiced segment 1: 0.2s - 1.0s
    mask1 = (t >= 0.2) & (t < 1.0)
    signal[mask1] = 0.5 * np.sin(2 * np.pi * f0 * t[mask1])
    
    # Voiced segment 2: 1.3s - 2.0s  
    mask2 = (t >= 1.3) & (t < 2.0)
    signal[mask2] = 0.4 * np.sin(2 * np.pi * f0 * t[mask2])
    
    # Voiced segment 3: 2.3s - 2.8s
    mask3 = (t >= 2.3) & (t < 2.8)
    signal[mask3] = 0.6 * np.sin(2 * np.pi * f0 * t[mask3])
    
    # Add harmonics
    signal += 0.1 * np.sin(2 * np.pi * 2 * f0 * t) * (mask1 | mask2 | mask3).astype(float)
    signal += 0.05 * np.sin(2 * np.pi * 3 * f0 * t) * (mask1 | mask2 | mask3).astype(float)
    
    # Add small noise everywhere
    signal += 0.01 * np.random.randn(len(signal))
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return signal.astype(np.float32)


@pytest.fixture
def silent_signal(sample_rate):
    """Generate a near-silent signal."""
    duration = 1.0
    return np.random.randn(int(sample_rate * duration)) * 0.001


@pytest.fixture
def temp_audio_file(test_signal, sample_rate):
    """Create a temporary WAV file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, test_signal, sample_rate)
        yield f.name
    # Cleanup
    os.unlink(f.name)


# ============== VAD Tests ==============

class TestVADFeatures:
    """Tests for VAD feature extraction."""
    
    def test_extract_windows_shape(self, test_signal, sample_rate):
        """Test that extract_windows returns correct shape."""
        win_size = int(0.025 * sample_rate)  # 25ms
        hop_size = int(0.01 * sample_rate)   # 10ms
        
        windows = extract_windows(test_signal, win_size, hop_size)
        
        expected_frames = int((len(test_signal) - win_size) / hop_size)
        assert windows.shape[0] == expected_frames
        assert windows.shape[1] == win_size
    
    def test_extract_windows_empty_signal(self, sample_rate):
        """Test extract_windows with very short signal."""
        short_signal = np.random.randn(100)  # Too short for standard windows
        win_size = int(0.025 * sample_rate)
        hop_size = int(0.01 * sample_rate)
        
        windows = extract_windows(short_signal, win_size, hop_size)
        assert windows.shape[0] == 0  # Should return empty
    
    def test_eVAD_output_structure(self, test_signal, sample_rate):
        """Test that eVAD returns correct output structure."""
        result = eVAD(test_signal, sample_rate)
        
        expected_keys = [
            'Pause_labels', 'Pause_duration', 'Pause_segments', 'Pause_times',
            'Speech_labels', 'Speech_duration', 'Speech_segments', 'Speech_times'
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_eVAD_labels_length(self, test_signal, sample_rate):
        """Test that VAD labels have correct length."""
        result = eVAD(test_signal, sample_rate)
        
        assert len(result['Pause_labels']) == len(test_signal)
        assert len(result['Speech_labels']) == len(test_signal)
    
    def test_eVAD_labels_binary(self, test_signal, sample_rate):
        """Test that VAD labels are binary."""
        result = eVAD(test_signal, sample_rate)
        
        assert set(np.unique(result['Pause_labels'])).issubset({0, 1})
        assert set(np.unique(result['Speech_labels'])).issubset({0, 1})
    
    def test_duration_feats_output_length(self, test_signal, sample_rate):
        """Test that duration_feats returns correct number of features."""
        vad_result = eVAD(test_signal, sample_rate)
        feats = duration_feats(vad_result, test_signal, sample_rate)
        
        assert len(feats) == 15  # 3 ratios + 6 speech stats + 6 pause stats
    
    def test_extract_vad_features_output(self, test_signal, sample_rate):
        """Test the main VAD feature extraction function."""
        features = extract_vad_features(test_signal, sample_rate)
        
        expected_features = get_vad_feature_names()
        for feat_name in expected_features:
            assert feat_name in features, f"Missing feature: {feat_name}"
            assert isinstance(features[feat_name], float), f"{feat_name} should be float"
    
    def test_vad_ratios_valid_range(self, test_signal, sample_rate):
        """Test that ratios are in valid ranges."""
        features = extract_vad_features(test_signal, sample_rate)
        
        # Ratios should be between 0 and 1
        assert 0 <= features['pause_ratio'] <= 1
        assert 0 <= features['speech_ratio'] <= 1
    
    def test_vad_silent_signal(self, silent_signal, sample_rate):
        """Test VAD on near-silent signal."""
        features = extract_vad_features(silent_signal, sample_rate)
        
        # Should still return valid features (no crashes)
        assert 'pause_ratio' in features
        assert not np.isnan(features['pause_ratio'])


# ============== Prosody Tests ==============

class TestProsodyFeatures:
    """Tests for prosody feature extraction."""
    
    def test_extract_f0_pyin(self, test_signal, sample_rate):
        """Test F0 extraction with pYIN."""
        f0 = extract_f0(test_signal, sample_rate, method='pyin')
        
        assert len(f0) > 0
        assert f0.dtype == np.float64 or f0.dtype == np.float32
    
    def test_extract_f0_yin(self, test_signal, sample_rate):
        """Test F0 extraction with YIN."""
        f0 = extract_f0(test_signal, sample_rate, method='yin')
        
        assert len(f0) > 0
    
    def test_f0_voiced_frames(self, test_signal, sample_rate):
        """Test that F0 has some voiced frames."""
        f0 = extract_f0(test_signal, sample_rate)
        
        voiced_frames = np.sum(f0 > 0)
        assert voiced_frames > 0, "Should detect some voiced frames"
    
    def test_extract_intensity(self, test_signal, sample_rate):
        """Test intensity extraction."""
        intensity = extract_intensity(test_signal, sample_rate)
        
        assert len(intensity) > 0
        assert np.isfinite(intensity).all(), "Intensity should be finite"
    
    def test_compute_f0_statistics(self, test_signal, sample_rate):
        """Test F0 statistics computation."""
        f0 = extract_f0(test_signal, sample_rate)
        stats = compute_f0_statistics(f0)
        
        expected_keys = ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range']
        for key in expected_keys:
            assert key in stats
    
    def test_f0_range_positive(self, test_signal, sample_rate):
        """Test that F0 range is non-negative."""
        features = extract_prosody_features(test_signal, sample_rate)
        
        assert features['f0_range'] >= 0
        assert features['f0_max'] >= features['f0_min']
    
    def test_extract_prosody_features_complete(self, test_signal, sample_rate):
        """Test complete prosody feature extraction."""
        features = extract_prosody_features(test_signal, sample_rate)
        
        expected_features = get_prosody_feature_names()
        for feat_name in expected_features:
            assert feat_name in features, f"Missing feature: {feat_name}"
    
    def test_voice_rate_positive(self, test_signal, sample_rate):
        """Test that voice rate is non-negative."""
        features = extract_prosody_features(test_signal, sample_rate)
        
        assert features['voice_rate'] >= 0


# ============== Voice Quality Tests ==============

class TestVoiceQualityFeatures:
    """Tests for voice quality feature extraction."""
    
    def test_compute_jitter_output(self, test_signal, sample_rate):
        """Test jitter computation."""
        jitter = compute_jitter(test_signal, sample_rate, use_parselmouth=False)
        
        assert 'jitter_mean' in jitter
        assert 'jitter_std' in jitter
        assert jitter['jitter_mean'] >= 0
    
    def test_compute_shimmer_output(self, test_signal, sample_rate):
        """Test shimmer computation."""
        shimmer = compute_shimmer(test_signal, sample_rate, use_parselmouth=False)
        
        assert 'shimmer_mean' in shimmer
        assert 'shimmer_std' in shimmer
        assert shimmer['shimmer_mean'] >= 0
    
    def test_compute_hnr_output(self, test_signal, sample_rate):
        """Test HNR computation."""
        hnr = compute_hnr(test_signal, sample_rate, use_parselmouth=False)
        
        assert 'hnr_mean' in hnr
        assert 'hnr_std' in hnr
        assert 'hnr_min' in hnr
        assert 'hnr_max' in hnr
    
    def test_extract_voice_quality_complete(self, test_signal, sample_rate):
        """Test complete voice quality extraction."""
        features = extract_voice_quality_features(
            test_signal, sample_rate, use_parselmouth=False
        )
        
        expected_features = get_voice_quality_feature_names()
        for feat_name in expected_features:
            assert feat_name in features, f"Missing feature: {feat_name}"


# ============== Audio Loading Tests ==============

class TestAudioLoading:
    """Tests for audio loading utilities."""
    
    def test_load_audio_wav(self, temp_audio_file, sample_rate):
        """Test loading WAV file."""
        signal, sr = load_audio(temp_audio_file, target_sr=sample_rate)
        
        assert sr == sample_rate
        assert len(signal) > 0
        assert signal.ndim == 1  # Mono
    
    def test_load_audio_resample(self, temp_audio_file):
        """Test resampling during load."""
        signal, sr = load_audio(temp_audio_file, target_sr=8000)
        
        assert sr == 8000
    
    def test_preprocess_audio_normalization(self, test_signal):
        """Test audio preprocessing normalization."""
        # Add DC offset
        signal_with_dc = test_signal + 0.5
        
        processed = preprocess_audio(signal_with_dc)
        
        # DC should be removed
        assert abs(np.mean(processed)) < 0.01
        # Should be normalized
        assert np.max(np.abs(processed)) <= 1.0
    
    def test_load_audio_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_audio('/nonexistent/path/audio.wav')


# ============== Validation Tests ==============

class TestValidation:
    """Tests for feature validation."""
    
    def test_validate_features_valid(self):
        """Test validation with valid features."""
        features = {
            'pause_ratio': 0.3,
            'speech_ratio': 0.7,
            'f0_mean': 150.0,
            'jitter_mean': 0.01
        }
        
        is_valid, issues = validate_features(features)
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_features_nan(self):
        """Test validation detects NaN."""
        features = {
            'pause_ratio': float('nan'),
            'speech_ratio': 0.7
        }
        
        _, issues = validate_features(features)
        assert any('NaN' in issue for issue in issues)
    
    def test_validate_features_out_of_range(self):
        """Test validation detects out-of-range values."""
        features = {
            'pause_ratio': 1.5,  # Should be <= 1.0
            'f0_mean': 1000.0  # Too high
        }
        
        _, issues = validate_features(features)
        assert len(issues) > 0


# ============== Integration Tests ==============

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_feature_extraction(self, test_signal, sample_rate):
        """Test extracting all features from a signal."""
        # VAD
        vad_feats = extract_vad_features(test_signal, sample_rate)
        
        # Prosody
        prosody_feats = extract_prosody_features(test_signal, sample_rate)
        
        # Voice quality
        vq_feats = extract_voice_quality_features(
            test_signal, sample_rate, use_parselmouth=False
        )
        
        # Combine
        all_feats = {**vad_feats, **prosody_feats, **vq_feats}
        
        # Validate
        is_valid, issues = validate_features(all_feats)
        
        # Should have all features
        total_expected = (
            len(get_vad_feature_names()) +
            len(get_prosody_feature_names()) +
            len(get_voice_quality_feature_names())
        )
        assert len(all_feats) == total_expected
    
    def test_feature_names_consistent(self):
        """Test that feature name lists are consistent."""
        vad_names = get_vad_feature_names()
        prosody_names = get_prosody_feature_names()
        vq_names = get_voice_quality_feature_names()
        
        # No duplicates
        all_names = vad_names + prosody_names + vq_names
        assert len(all_names) == len(set(all_names))
        
        # All have valid ranges defined
        for name in all_names:
            if name in FEATURE_RANGES:
                min_val, max_val = FEATURE_RANGES[name]
                assert min_val < max_val


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

