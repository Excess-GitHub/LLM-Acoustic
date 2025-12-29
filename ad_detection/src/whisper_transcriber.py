"""
Whisper-based automatic speech recognition for audio transcription.

Used to generate transcripts from audio files when ground-truth
transcripts are not available or for comparison.
"""

import os
from typing import Optional, List, Dict, Union
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm


class WhisperTranscriber:
    """
    Whisper ASR for transcribing audio files.
    
    The paper uses OpenAI Whisper to transcribe audio recordings
    before feeding to the LLM.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: str = "en",
        task: str = "transcribe"
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3').
            device: Device to run on ('cuda', 'cpu', or None for auto).
            language: Language code.
            task: Task type ('transcribe' or 'translate').
        """
        self.model_size = model_size
        self.language = language
        self.task = task
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            import whisper
            print(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"Whisper model loaded on {self.device}")
        except ImportError:
            print("Warning: openai-whisper not installed. Install with: pip install openai-whisper")
            raise
    
    def transcribe(
        self,
        audio_path: str,
        **kwargs
    ) -> Dict:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file.
            **kwargs: Additional arguments for whisper.transcribe().
            
        Returns:
            Dictionary with 'text', 'segments', and 'language'.
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            task=self.task,
            **kwargs
        )
        
        return {
            'text': result['text'].strip(),
            'segments': result.get('segments', []),
            'language': result.get('language', self.language)
        }
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths.
            show_progress: Show progress bar.
            **kwargs: Additional arguments for whisper.transcribe().
            
        Returns:
            List of transcription dictionaries.
        """
        results = []
        
        iterator = tqdm(audio_paths, desc="Transcribing") if show_progress else audio_paths
        
        for audio_path in iterator:
            try:
                result = self.transcribe(audio_path, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error transcribing {audio_path}: {e}")
                results.append({
                    'text': '',
                    'segments': [],
                    'language': self.language,
                    'error': str(e)
                })
        
        return results
    
    def transcribe_to_file(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio and save to text file.
        
        Args:
            audio_path: Path to audio file.
            output_path: Path to save transcript (default: same name with .txt).
            **kwargs: Additional arguments for whisper.transcribe().
            
        Returns:
            Path to output file.
        """
        if output_path is None:
            output_path = str(Path(audio_path).with_suffix('.txt'))
        
        result = self.transcribe(audio_path, **kwargs)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        return output_path


class FasterWhisperTranscriber:
    """
    Faster-Whisper based transcriber for better performance.
    
    Uses CTranslate2 for faster inference.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float16",
        language: str = "en"
    ):
        """
        Initialize Faster-Whisper transcriber.
        
        Args:
            model_size: Model size.
            device: Device ('cuda', 'cpu', or 'auto').
            compute_type: Compute type ('float16', 'int8', 'float32').
            language: Language code.
        """
        self.model_size = model_size
        self.language = language
        self.compute_type = compute_type
        
        if device is None or device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Faster-Whisper model."""
        try:
            from faster_whisper import WhisperModel
            print(f"Loading Faster-Whisper {self.model_size} model...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            print(f"Faster-Whisper model loaded on {self.device}")
        except ImportError:
            print("Warning: faster-whisper not installed. Install with: pip install faster-whisper")
            raise
    
    def transcribe(
        self,
        audio_path: str,
        **kwargs
    ) -> Dict:
        """Transcribe a single audio file."""
        if self.model is None:
            raise RuntimeError("Faster-Whisper model not loaded")
        
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            **kwargs
        )
        
        # Collect all segments
        all_segments = list(segments)
        text = ' '.join([seg.text.strip() for seg in all_segments])
        
        return {
            'text': text,
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text
                }
                for seg in all_segments
            ],
            'language': info.language
        }
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict]:
        """Transcribe multiple audio files."""
        results = []
        iterator = tqdm(audio_paths, desc="Transcribing") if show_progress else audio_paths
        
        for audio_path in iterator:
            try:
                result = self.transcribe(audio_path, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error transcribing {audio_path}: {e}")
                results.append({
                    'text': '',
                    'segments': [],
                    'error': str(e)
                })
        
        return results


def get_transcriber(
    backend: str = "whisper",
    **kwargs
) -> Union[WhisperTranscriber, FasterWhisperTranscriber]:
    """
    Get appropriate transcriber based on backend.
    
    Args:
        backend: 'whisper' or 'faster-whisper'.
        **kwargs: Arguments for transcriber.
        
    Returns:
        Transcriber instance.
    """
    if backend == "whisper":
        return WhisperTranscriber(**kwargs)
    elif backend == "faster-whisper":
        return FasterWhisperTranscriber(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    # Test transcription
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    transcriber = WhisperTranscriber(
        model_size=config['whisper']['model_size'],
        language=config['whisper']['language']
    )
    
    # Test with a sample file
    test_audio = "../Pitt Corpus/Pitt Corpus/Media/Dementia/Cookie/WAV/001-0.wav"
    if os.path.exists(test_audio):
        result = transcriber.transcribe(test_audio)
        print(f"\nTranscription:\n{result['text']}")

