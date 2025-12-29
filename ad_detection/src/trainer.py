"""
Main trainer for AD detection system.

Orchestrates the full pipeline:
1. Load data from Pitt Corpus
2. Transcribe audio with Whisper (if needed)
3. Extract LLM features (fine-tune if needed)
4. Extract acoustic features (VGGish + GRU-AE)
5. Fuse features and select important ones
6. Train classifier
7. Evaluate with cross-validation
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .data_loader import PittCorpusDataset, PittSample, load_pitt_corpus
from .whisper_transcriber import WhisperTranscriber, get_transcriber
from .llm_features import LLMFeatureExtractor, create_llm_extractor
from .vggish_features import VGGishFeatureExtractor
from .gru_autoencoder import GRUAutoencoder, GRUAutoencoderTrainer, VGGishDataset, create_gru_autoencoder
from .feature_fusion import FeatureFusion, CustomAcousticFusion, create_feature_fusion
from .classifier import ADClassifier, create_classifier, evaluate_with_repeats


class ADDetectionTrainer:
    """
    Main trainer for AD detection system.
    
    Implements the full LLM-A-X pipeline from the paper.
    """
    
    def __init__(
        self,
        config: dict,
        small_gpu: bool = False,
        use_custom_acoustic: bool = False,
        output_dir: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary.
            small_gpu: Use small GPU optimizations.
            use_custom_acoustic: Use custom acoustic features instead of VGGish.
            output_dir: Output directory for results.
        """
        self.config = config
        self.small_gpu = small_gpu
        self.use_custom_acoustic = use_custom_acoustic
        
        self.output_dir = output_dir or config['paths'].get('output_dir', './output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.dataset = None
        self.transcriber = None
        self.llm_extractor = None
        self.vggish_extractor = None
        self.gru_autoencoder = None
        self.feature_fusion = None
        self.classifier = None
        
        # Cache for extracted features
        self.llm_features_cache = {}
        self.acoustic_features_cache = {}
    
    def load_data(self):
        """Load the Pitt Corpus dataset."""
        print("Loading Pitt Corpus dataset...")
        
        self.dataset = PittCorpusDataset(
            corpus_root=self.config['paths']['pitt_corpus_root'],
            custom_features_csv=self.config['paths'].get('custom_audio_features'),
            task='cookie',
            random_seed=self.config['dataset'].get('random_seed', 42)
        )
        
        return self
    
    def setup_transcriber(self):
        """Setup Whisper transcriber."""
        whisper_config = self.config.get('whisper', {})
        
        self.transcriber = WhisperTranscriber(
            model_size=whisper_config.get('model_size', 'base'),
            language=whisper_config.get('language', 'en')
        )
        
        return self
    
    def get_transcripts(
        self,
        samples: List[PittSample],
        use_whisper: bool = False
    ) -> List[str]:
        """
        Get transcripts for samples.
        
        Args:
            samples: List of samples.
            use_whisper: Use Whisper ASR instead of ground truth transcripts.
            
        Returns:
            List of transcript strings.
        """
        transcripts = []
        
        if use_whisper:
            if self.transcriber is None:
                self.setup_transcriber()
            
            audio_paths = [s.audio_path for s in samples]
            results = self.transcriber.transcribe_batch(audio_paths)
            transcripts = [r['text'] for r in results]
        else:
            # Use ground truth transcripts from CHAT files
            transcripts = [s.transcript_text or '' for s in samples]
        
        return transcripts
    
    def setup_llm_extractor(self):
        """Setup LLM feature extractor."""
        print("Setting up LLM feature extractor...")
        
        self.llm_extractor = create_llm_extractor(
            self.config,
            small_gpu=self.small_gpu
        )
        self.llm_extractor.load_model()
        
        return self
    
    def setup_vggish_extractor(self):
        """Setup VGGish feature extractor."""
        print("Setting up VGGish feature extractor...")
        
        vggish_config = self.config.get('vggish', {})
        
        self.vggish_extractor = VGGishFeatureExtractor(
            sample_rate=vggish_config.get('sample_rate', 16000)
        )
        
        return self
    
    def _clear_gpu_memory(self):
        """Clear GPU memory to free up VRAM for next operation."""
        import gc
        
        # Delete LLM model from GPU if loaded
        if self.llm_extractor is not None:
            if hasattr(self.llm_extractor, 'model') and self.llm_extractor.model is not None:
                del self.llm_extractor.model
                self.llm_extractor.model = None
            if hasattr(self.llm_extractor, 'peft_model') and self.llm_extractor.peft_model is not None:
                del self.llm_extractor.peft_model
                self.llm_extractor.peft_model = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Report memory
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU memory cleared: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def setup_gru_autoencoder(self):
        """Setup GRU autoencoder."""
        print("Setting up GRU autoencoder...")
        
        self.gru_autoencoder = create_gru_autoencoder(self.config)
        
        return self
    
    def extract_llm_features(
        self,
        samples: List[PittSample],
        fine_tune: bool = True,
        labels: Optional[List[int]] = None,
        use_cache: bool = True,
        fold_idx: int = 0
    ) -> np.ndarray:
        """
        Extract LLM features from samples.
        
        Args:
            samples: List of samples.
            fine_tune: Whether to fine-tune the LLM first.
            labels: Labels for fine-tuning.
            use_cache: Use cached features if available.
            fold_idx: Fold index for fold-specific checkpoints.
            
        Returns:
            Feature array (n_samples, hidden_dim).
        """
        if self.llm_extractor is None:
            self.setup_llm_extractor()
        
        # Get transcripts
        transcripts = self.get_transcripts(samples, use_whisper=False)
        
        # Check cache
        cache_key = hash(tuple(transcripts))
        if use_cache and cache_key in self.llm_features_cache:
            print("Using cached LLM features")
            return self.llm_features_cache[cache_key]
        
        # Fine-tune if needed (with checkpoint resume support)
        # Use fold-specific checkpoint directory to avoid cross-fold contamination
        if fine_tune and labels is not None and not self.llm_extractor.is_fine_tuned:
            llm_checkpoint_dir = os.path.join(self.output_dir, 'llm_checkpoints', f'fold_{fold_idx}', 'final')
            
            # Check if we have a saved checkpoint to resume from
            if os.path.exists(llm_checkpoint_dir) and os.path.exists(os.path.join(llm_checkpoint_dir, 'adapter_config.json')):
                print(f"Found existing fine-tuned model at {llm_checkpoint_dir}")
                print("Loading saved checkpoint (skipping fine-tuning)...")
                try:
                    self.llm_extractor.load(llm_checkpoint_dir)
                    print("Successfully loaded fine-tuned model!")
                except Exception as e:
                    print(f"Warning: Could not load checkpoint: {e}")
                    print("Will re-run fine-tuning...")
                    self._run_fine_tuning(transcripts, labels, fold_idx)
            else:
                self._run_fine_tuning(transcripts, labels, fold_idx)
        
        # Extract features
        print("Extracting LLM features...")
        features = self.llm_extractor.extract_features(transcripts)
        
        # Cache
        if use_cache:
            self.llm_features_cache[cache_key] = features
        
        return features
    
    def _run_fine_tuning(self, transcripts: List[str], labels: List[int], fold_idx: int = 0):
        """Run LLM fine-tuning.
        
        Args:
            transcripts: List of transcripts to train on.
            labels: Training labels.
            fold_idx: Fold index for fold-specific checkpoint directory.
        """
        print("Fine-tuning LLM...")
        
        train_config = self.config.get('training', {})
        
        if self.small_gpu:
            small_config = self.config.get('small_gpu', {})
            batch_size = small_config.get('batch_size', 1)
            grad_accum = small_config.get('gradient_accumulation_steps', 16)
            learning_rate = 2e-5  # Slightly higher for smaller model
            print(f"  Small GPU mode: batch_size={batch_size}, grad_accum={grad_accum}")
        else:
            batch_size = train_config.get('batch_size', 2)
            grad_accum = train_config.get('gradient_accumulation_steps', 8)
            learning_rate = train_config.get('learning_rate', 1e-5)
        
        # Use fold-specific checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, 'llm_checkpoints', f'fold_{fold_idx}')
        
        self.llm_extractor.fine_tune(
            train_transcripts=transcripts,
            train_labels=labels,
            epochs=train_config.get('epochs', 5),
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            output_dir=checkpoint_dir,
            gradient_checkpointing=self.small_gpu
        )
    
    def extract_vggish_features(
        self,
        samples: List[PittSample],
        use_cache: bool = True
    ) -> List[np.ndarray]:
        """
        Extract VGGish features from audio files.
        
        Args:
            samples: List of samples.
            use_cache: Use cached features.
            
        Returns:
            List of VGGish feature arrays (variable length sequences).
        """
        if self.vggish_extractor is None:
            self.setup_vggish_extractor()
        
        # Check cache
        cache_key = hash(tuple(s.audio_path for s in samples))
        if use_cache and cache_key in self.acoustic_features_cache:
            print("Using cached VGGish features")
            return self.acoustic_features_cache[cache_key]
        
        print("Extracting VGGish features...")
        audio_paths = [s.audio_path for s in samples]
        features = self.vggish_extractor.extract_batch(audio_paths)
        
        if use_cache:
            self.acoustic_features_cache[cache_key] = features
        
        return features
    
    def train_gru_autoencoder(
        self,
        vggish_features: List[np.ndarray],
        labels: List[int]
    ) -> np.ndarray:
        """
        Train GRU autoencoder and extract fixed-dim acoustic features.
        
        Args:
            vggish_features: List of VGGish feature sequences.
            labels: Labels for supervised training.
            
        Returns:
            Fixed-dimension acoustic features (n_samples, feature_dim).
        """
        if self.gru_autoencoder is None:
            self.setup_gru_autoencoder()
        
        gru_config = self.config.get('gru_autoencoder', {})
        device = self.config.get('training', {}).get('device', 'cuda')
        
        # Create dataset
        dataset = VGGishDataset(vggish_features, labels)
        train_loader = DataLoader(
            dataset,
            batch_size=gru_config.get('batch_size', 16),
            shuffle=True
        )
        
        # Train
        print("Training GRU autoencoder...")
        trainer = GRUAutoencoderTrainer(
            model=self.gru_autoencoder,
            reconstruction_weight=gru_config.get('reconstruction_weight', 0.5),
            learning_rate=gru_config.get('learning_rate', 0.001),
            device=device
        )
        
        trainer.train(
            train_loader=train_loader,
            epochs=gru_config.get('epochs', 50),
            patience=10
        )
        
        # Extract features
        print("Extracting acoustic features from GRU-AE...")
        acoustic_features = trainer.extract_features(vggish_features)
        
        return acoustic_features
    
    def get_custom_acoustic_features(
        self,
        samples: List[PittSample]
    ) -> np.ndarray:
        """
        Get custom acoustic features from the audio_feature_extraction pipeline.
        
        Args:
            samples: List of samples.
            
        Returns:
            Custom acoustic features array.
        """
        features = []
        
        for sample in samples:
            custom_feat = self.dataset.get_custom_features(sample)
            if custom_feat is not None:
                features.append(custom_feat)
            else:
                # Use zeros as fallback
                print(f"Warning: No custom features for {sample.participant_id}-{sample.session}")
                features.append(np.zeros(42))  # 42 features in custom extraction
        
        return np.vstack(features)
    
    def train_fold(
        self,
        train_samples: List[PittSample],
        test_samples: List[PittSample],
        fold_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Train and evaluate on a single fold.
        
        Args:
            train_samples: Training samples.
            test_samples: Test samples.
            fold_idx: Fold index for logging.
            
        Returns:
            Dictionary with results.
        """
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}")
        print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
        print(f"{'='*60}")
        
        # Get labels
        train_labels = [1 if s.group == 'AD' else 0 for s in train_samples]
        test_labels = [1 if s.group == 'AD' else 0 for s in test_samples]
        
        # Extract LLM features (use fold-specific checkpoints)
        train_llm_features = self.extract_llm_features(
            train_samples,
            fine_tune=True,
            labels=train_labels,
            use_cache=False,  # Don't cache during CV
            fold_idx=fold_idx
        )
        
        test_llm_features = self.extract_llm_features(
            test_samples,
            fine_tune=False,
            use_cache=False,
            fold_idx=fold_idx
        )
        
        # Extract acoustic features
        if self.use_custom_acoustic:
            print("Using custom acoustic features...")
            train_acoustic = self.get_custom_acoustic_features(train_samples)
            test_acoustic = self.get_custom_acoustic_features(test_samples)
        else:
            print("Using VGGish + GRU-AE features...")
            
            # Clear GPU memory before VGGish extraction
            self._clear_gpu_memory()
            
            train_vggish = self.extract_vggish_features(train_samples, use_cache=False)
            test_vggish = self.extract_vggish_features(test_samples, use_cache=False)
            
            # Train GRU-AE and extract features
            train_acoustic = self.train_gru_autoencoder(train_vggish, train_labels)
            
            # Extract test features using trained encoder
            gru_trainer = GRUAutoencoderTrainer(
                self.gru_autoencoder,
                device=self.config.get('training', {}).get('device', 'cuda')
            )
            test_acoustic = gru_trainer.extract_features(test_vggish)
        
        # Feature fusion
        print("Fusing features...")
        self.feature_fusion = create_feature_fusion(
            self.config,
            use_custom_acoustic=self.use_custom_acoustic
        )
        
        train_fused = self.feature_fusion.fit_transform(
            train_llm_features,
            train_acoustic,
            np.array(train_labels)
        )
        test_fused = self.feature_fusion.transform(test_llm_features, test_acoustic)
        
        # Train classifier
        print("Training classifier...")
        self.classifier = create_classifier(self.config)
        self.classifier.fit(train_fused, np.array(train_labels))
        
        # Evaluate
        results = self.classifier.evaluate(test_fused, np.array(test_labels))
        
        print(f"\nFold {fold_idx + 1} Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        # Get feature importance breakdown
        importance_info = self.feature_fusion.get_feature_importance_breakdown() \
            if hasattr(self.feature_fusion, 'get_feature_importance_breakdown') else {}
        
        return {
            'fold': fold_idx + 1,
            'train_size': len(train_samples),
            'test_size': len(test_samples),
            'metrics': results,
            'feature_importance': importance_info
        }
    
    def run_cross_validation(
        self,
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Run full cross-validation.
        
        Args:
            n_folds: Number of folds.
            
        Returns:
            Dictionary with all results.
        """
        if self.dataset is None:
            self.load_data()
        
        folds = self.dataset.get_cv_folds(n_folds=n_folds)
        
        all_results = []
        
        for fold_idx, (train_samples, test_samples) in enumerate(folds):
            fold_results = self.train_fold(train_samples, test_samples, fold_idx)
            all_results.append(fold_results)
            
            # Reset for next fold
            self.llm_extractor.is_fine_tuned = False
            self.gru_autoencoder = None
        
        # Aggregate results
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        summary = {}
        
        for metric in metrics:
            values = [r['metrics'].get(metric, 0) for r in all_results]
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        # Print summary
        print(f"\n{'='*60}")
        print("Cross-Validation Summary")
        print(f"{'='*60}")
        for metric, stats in summary.items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Save results
        results = {
            'config': {
                'small_gpu': self.small_gpu,
                'use_custom_acoustic': self.use_custom_acoustic,
                'n_folds': n_folds
            },
            'folds': all_results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        
        output_path = os.path.join(
            self.output_dir,
            f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        
        return results
    
    def train_final_model(
        self,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train final model on all data.
        
        Args:
            save_path: Path to save the trained model.
            
        Returns:
            Training results.
        """
        if self.dataset is None:
            self.load_data()
        
        all_samples = self.dataset.samples
        labels = [1 if s.group == 'AD' else 0 for s in all_samples]
        
        print(f"Training final model on {len(all_samples)} samples...")
        
        # Extract all features
        llm_features = self.extract_llm_features(
            all_samples,
            fine_tune=True,
            labels=labels
        )
        
        if self.use_custom_acoustic:
            acoustic_features = self.get_custom_acoustic_features(all_samples)
        else:
            vggish_features = self.extract_vggish_features(all_samples)
            acoustic_features = self.train_gru_autoencoder(vggish_features, labels)
        
        # Fuse and select features
        self.feature_fusion = create_feature_fusion(
            self.config,
            use_custom_acoustic=self.use_custom_acoustic
        )
        fused_features = self.feature_fusion.fit_transform(
            llm_features,
            acoustic_features,
            np.array(labels)
        )
        
        # Train classifier
        self.classifier = create_classifier(self.config)
        self.classifier.fit(fused_features, np.array(labels))
        
        # Save model
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'final_model')
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save components
        self.llm_extractor.save(os.path.join(save_path, 'llm'))
        
        if self.gru_autoencoder is not None:
            torch.save(
                self.gru_autoencoder.state_dict(),
                os.path.join(save_path, 'gru_autoencoder.pt')
            )
        
        with open(os.path.join(save_path, 'feature_fusion.pkl'), 'wb') as f:
            pickle.dump(self.feature_fusion, f)
        
        with open(os.path.join(save_path, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        
        print(f"Model saved to: {save_path}")
        
        return {
            'n_samples': len(all_samples),
            'feature_dim': fused_features.shape[1],
            'model_path': save_path
        }
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        # Load LLM
        self.setup_llm_extractor()
        self.llm_extractor.load(os.path.join(model_path, 'llm'))
        
        # Load GRU-AE
        gru_path = os.path.join(model_path, 'gru_autoencoder.pt')
        if os.path.exists(gru_path):
            self.setup_gru_autoencoder()
            self.gru_autoencoder.load_state_dict(torch.load(gru_path))
        
        # Load feature fusion
        with open(os.path.join(model_path, 'feature_fusion.pkl'), 'rb') as f:
            self.feature_fusion = pickle.load(f)
        
        # Load classifier
        with open(os.path.join(model_path, 'classifier.pkl'), 'rb') as f:
            self.classifier = pickle.load(f)
        
        print(f"Model loaded from: {model_path}")
        
        return self
    
    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict AD for a single audio file.
        
        Args:
            audio_path: Path to audio file.
            transcript: Optional transcript (uses Whisper if not provided).
            
        Returns:
            Prediction results.
        """
        # Get transcript
        if transcript is None:
            if self.transcriber is None:
                self.setup_transcriber()
            result = self.transcriber.transcribe(audio_path)
            transcript = result['text']
        
        # Extract features
        llm_features = self.llm_extractor.extract_features([transcript])
        
        if self.use_custom_acoustic:
            raise NotImplementedError("Custom acoustic features require Pitt Corpus sample")
        else:
            vggish_features = self.vggish_extractor.extract(audio_path)
            
            # Use trained GRU-AE
            device = self.config.get('training', {}).get('device', 'cuda')
            trainer = GRUAutoencoderTrainer(self.gru_autoencoder, device=device)
            acoustic_features = trainer.extract_features([vggish_features])
        
        # Fuse features
        fused_features = self.feature_fusion.transform(llm_features, acoustic_features)
        
        # Predict
        prediction = self.classifier.predict(fused_features)[0]
        probabilities = self.classifier.predict_proba(fused_features)[0]
        
        return {
            'prediction': 'AD' if prediction == 1 else 'HC',
            'confidence': float(max(probabilities)),
            'probabilities': {
                'HC': float(probabilities[0]),
                'AD': float(probabilities[1])
            },
            'transcript': transcript
        }


def create_trainer(
    config_path: str,
    small_gpu: bool = False,
    use_custom_acoustic: bool = False
) -> ADDetectionTrainer:
    """Create trainer from config file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return ADDetectionTrainer(
        config=config,
        small_gpu=small_gpu,
        use_custom_acoustic=use_custom_acoustic
    )

