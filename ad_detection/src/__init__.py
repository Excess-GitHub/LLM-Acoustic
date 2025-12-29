"""
AD Detection with Fine-Tuned LLM and Acoustic Features

Based on: "Integrating Fine-Tuned LLM with Acoustic Features for Enhanced Detection 
of Alzheimer's Disease" by Casu et al.

Best model configuration:
- LLM: Mistral-7B (fine-tuned with LoRA)
- Acoustic: VGGish + GRU Autoencoder
- Classifier: SVC with RBF kernel
"""

from .data_loader import PittCorpusDataset, load_pitt_corpus
from .whisper_transcriber import WhisperTranscriber
from .llm_features import LLMFeatureExtractor
from .vggish_features import VGGishFeatureExtractor
from .gru_autoencoder import GRUAutoencoder
from .feature_fusion import FeatureFusion
from .classifier import ADClassifier
from .trainer import ADDetectionTrainer

__version__ = "1.0.0"
__all__ = [
    "PittCorpusDataset",
    "load_pitt_corpus", 
    "WhisperTranscriber",
    "LLMFeatureExtractor",
    "VGGishFeatureExtractor",
    "GRUAutoencoder",
    "FeatureFusion",
    "ADClassifier",
    "ADDetectionTrainer",
]

