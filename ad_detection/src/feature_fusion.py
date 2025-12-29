"""
Feature fusion and selection for multimodal AD detection.

Based on paper Section III-D:
1. Concatenate LLM features (4096-dim) with acoustic features (64-dim from GRU-AE)
2. Apply feature selection using LinearSVC with L1 penalty
3. Select features accounting for 95% of total importance
"""

import numpy as np
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


class FeatureFusion:
    """
    Multimodal feature fusion with dimensionality reduction.
    
    Combines linguistic (LLM) and acoustic features, then selects
    the most important features using L1-regularized SVC.
    """
    
    def __init__(
        self,
        importance_threshold: float = 0.95,
        normalize: bool = True,
        random_state: int = 42
    ):
        """
        Initialize feature fusion.
        
        Args:
            importance_threshold: Cumulative importance threshold (0.95 = keep 95%).
            normalize: Whether to standardize features before fusion.
            random_state: Random state for reproducibility.
        """
        self.importance_threshold = importance_threshold
        self.normalize = normalize
        self.random_state = random_state
        
        # Scalers
        self.llm_scaler = StandardScaler() if normalize else None
        self.acoustic_scaler = StandardScaler() if normalize else None
        
        # Feature selector
        self.selector = None
        self.feature_importances_ = None
        self.selected_indices_ = None
        self.n_llm_features_ = None
        self.n_acoustic_features_ = None
    
    def fit(
        self,
        llm_features: np.ndarray,
        acoustic_features: np.ndarray,
        labels: np.ndarray
    ) -> 'FeatureFusion':
        """
        Fit the feature fusion pipeline.
        
        Args:
            llm_features: LLM feature array (n_samples, llm_dim).
            acoustic_features: Acoustic feature array (n_samples, acoustic_dim).
            labels: Binary labels (0=HC, 1=AD).
            
        Returns:
            self
        """
        self.n_llm_features_ = llm_features.shape[1]
        self.n_acoustic_features_ = acoustic_features.shape[1]
        
        # Normalize features
        if self.normalize:
            llm_features = self.llm_scaler.fit_transform(llm_features)
            acoustic_features = self.acoustic_scaler.fit_transform(acoustic_features)
        
        # Concatenate features
        combined = np.hstack([llm_features, acoustic_features])
        
        # Fit LinearSVC with L1 penalty for feature selection
        # From paper: "We employed the LinearSVC from Scikit-learn library, 
        # with an L1 penalty, to rank the features based on their importance"
        svc = LinearSVC(
            penalty='l1',
            dual=False,
            C=1.0,
            max_iter=10000,
            random_state=self.random_state
        )
        svc.fit(combined, labels)
        
        # Get feature importances (absolute values of coefficients)
        importances = np.abs(svc.coef_).flatten()
        
        # Normalize importances to [0, 1]
        if importances.max() > 0:
            importances = importances / importances.max()
        
        self.feature_importances_ = importances
        
        # Select features based on cumulative importance
        # Sort indices by importance (descending)
        sorted_indices = np.argsort(importances)[::-1]
        cumulative_importance = np.cumsum(importances[sorted_indices])
        total_importance = cumulative_importance[-1]
        
        # Find cutoff
        if total_importance > 0:
            normalized_cumulative = cumulative_importance / total_importance
            n_features = np.searchsorted(normalized_cumulative, self.importance_threshold) + 1
            n_features = min(n_features, len(sorted_indices))
        else:
            n_features = len(sorted_indices)
        
        self.selected_indices_ = sorted_indices[:n_features]
        
        print(f"Feature selection: {len(self.selected_indices_)} / {combined.shape[1]} features retained")
        print(f"  LLM features retained: {np.sum(self.selected_indices_ < self.n_llm_features_)}")
        print(f"  Acoustic features retained: {np.sum(self.selected_indices_ >= self.n_llm_features_)}")
        
        return self
    
    def transform(
        self,
        llm_features: np.ndarray,
        acoustic_features: np.ndarray
    ) -> np.ndarray:
        """
        Transform features using fitted pipeline.
        
        Args:
            llm_features: LLM feature array.
            acoustic_features: Acoustic feature array.
            
        Returns:
            Selected feature array.
        """
        if self.selected_indices_ is None:
            raise RuntimeError("FeatureFusion not fitted. Call fit() first.")
        
        # Normalize
        if self.normalize:
            llm_features = self.llm_scaler.transform(llm_features)
            acoustic_features = self.acoustic_scaler.transform(acoustic_features)
        
        # Concatenate
        combined = np.hstack([llm_features, acoustic_features])
        
        # Select features
        return combined[:, self.selected_indices_]
    
    def fit_transform(
        self,
        llm_features: np.ndarray,
        acoustic_features: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(llm_features, acoustic_features, labels)
        return self.transform(llm_features, acoustic_features)
    
    def get_feature_importance_breakdown(self) -> dict:
        """
        Get breakdown of feature importances by modality.
        
        Returns:
            Dictionary with importance statistics.
        """
        if self.feature_importances_ is None:
            return {}
        
        llm_importances = self.feature_importances_[:self.n_llm_features_]
        acoustic_importances = self.feature_importances_[self.n_llm_features_:]
        
        selected_llm = np.sum(self.selected_indices_ < self.n_llm_features_)
        selected_acoustic = np.sum(self.selected_indices_ >= self.n_llm_features_)
        
        return {
            'total_features': len(self.feature_importances_),
            'selected_features': len(self.selected_indices_),
            'llm_features': {
                'total': self.n_llm_features_,
                'selected': selected_llm,
                'mean_importance': float(llm_importances.mean()),
                'max_importance': float(llm_importances.max()),
                'sum_importance': float(llm_importances.sum())
            },
            'acoustic_features': {
                'total': self.n_acoustic_features_,
                'selected': selected_acoustic,
                'mean_importance': float(acoustic_importances.mean()),
                'max_importance': float(acoustic_importances.max()),
                'sum_importance': float(acoustic_importances.sum())
            }
        }


class LLMOnlyFusion:
    """
    Feature processor for LLM-only experiments (no acoustic features).
    """
    
    def __init__(
        self,
        importance_threshold: float = 0.95,
        normalize: bool = True,
        random_state: int = 42
    ):
        self.importance_threshold = importance_threshold
        self.normalize = normalize
        self.random_state = random_state
        
        self.scaler = StandardScaler() if normalize else None
        self.selected_indices_ = None
        self.feature_importances_ = None
    
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> 'LLMOnlyFusion':
        """Fit feature selection."""
        if self.normalize:
            features = self.scaler.fit_transform(features)
        
        # Feature selection
        svc = LinearSVC(
            penalty='l1',
            dual=False,
            C=1.0,
            max_iter=10000,
            random_state=self.random_state
        )
        svc.fit(features, labels)
        
        importances = np.abs(svc.coef_).flatten()
        if importances.max() > 0:
            importances = importances / importances.max()
        
        self.feature_importances_ = importances
        
        sorted_indices = np.argsort(importances)[::-1]
        cumulative_importance = np.cumsum(importances[sorted_indices])
        total_importance = cumulative_importance[-1]
        
        if total_importance > 0:
            normalized_cumulative = cumulative_importance / total_importance
            n_features = np.searchsorted(normalized_cumulative, self.importance_threshold) + 1
            n_features = min(n_features, len(sorted_indices))
        else:
            n_features = len(sorted_indices)
        
        self.selected_indices_ = sorted_indices[:n_features]
        
        print(f"Feature selection: {len(self.selected_indices_)} / {features.shape[1]} features retained")
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features."""
        if self.selected_indices_ is None:
            raise RuntimeError("Not fitted")
        
        if self.normalize:
            features = self.scaler.transform(features)
        
        return features[:, self.selected_indices_]
    
    def fit_transform(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """Fit and transform."""
        self.fit(features, labels)
        return self.transform(features)


class CustomAcousticFusion:
    """
    Feature fusion for custom acoustic features (from audio_feature_extraction).
    
    Handles the 42 acoustic features from the custom extraction pipeline.
    """
    
    def __init__(
        self,
        importance_threshold: float = 0.95,
        normalize: bool = True,
        random_state: int = 42
    ):
        self.importance_threshold = importance_threshold
        self.normalize = normalize
        self.random_state = random_state
        
        self.llm_scaler = StandardScaler() if normalize else None
        self.acoustic_scaler = StandardScaler() if normalize else None
        
        self.selected_indices_ = None
        self.feature_importances_ = None
        self.n_llm_features_ = None
        self.n_acoustic_features_ = None
    
    def fit(
        self,
        llm_features: np.ndarray,
        acoustic_features: np.ndarray,
        labels: np.ndarray
    ) -> 'CustomAcousticFusion':
        """Fit fusion pipeline."""
        self.n_llm_features_ = llm_features.shape[1]
        self.n_acoustic_features_ = acoustic_features.shape[1]
        
        # Handle NaN values in custom features
        acoustic_features = np.nan_to_num(acoustic_features, nan=0.0)
        
        if self.normalize:
            llm_features = self.llm_scaler.fit_transform(llm_features)
            acoustic_features = self.acoustic_scaler.fit_transform(acoustic_features)
        
        combined = np.hstack([llm_features, acoustic_features])
        
        # Feature selection
        svc = LinearSVC(
            penalty='l1',
            dual=False,
            C=1.0,
            max_iter=10000,
            random_state=self.random_state
        )
        svc.fit(combined, labels)
        
        importances = np.abs(svc.coef_).flatten()
        if importances.max() > 0:
            importances = importances / importances.max()
        
        self.feature_importances_ = importances
        
        sorted_indices = np.argsort(importances)[::-1]
        cumulative_importance = np.cumsum(importances[sorted_indices])
        total_importance = cumulative_importance[-1]
        
        if total_importance > 0:
            normalized_cumulative = cumulative_importance / total_importance
            n_features = np.searchsorted(normalized_cumulative, self.importance_threshold) + 1
            n_features = min(n_features, len(sorted_indices))
        else:
            n_features = len(sorted_indices)
        
        self.selected_indices_ = sorted_indices[:n_features]
        
        print(f"Custom Acoustic Fusion:")
        print(f"  Total features: {combined.shape[1]} -> {len(self.selected_indices_)} selected")
        print(f"  LLM: {self.n_llm_features_} -> {np.sum(self.selected_indices_ < self.n_llm_features_)}")
        print(f"  Acoustic: {self.n_acoustic_features_} -> {np.sum(self.selected_indices_ >= self.n_llm_features_)}")
        
        return self
    
    def transform(
        self,
        llm_features: np.ndarray,
        acoustic_features: np.ndarray
    ) -> np.ndarray:
        """Transform features."""
        if self.selected_indices_ is None:
            raise RuntimeError("Not fitted")
        
        acoustic_features = np.nan_to_num(acoustic_features, nan=0.0)
        
        if self.normalize:
            llm_features = self.llm_scaler.transform(llm_features)
            acoustic_features = self.acoustic_scaler.transform(acoustic_features)
        
        combined = np.hstack([llm_features, acoustic_features])
        return combined[:, self.selected_indices_]
    
    def fit_transform(
        self,
        llm_features: np.ndarray,
        acoustic_features: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """Fit and transform."""
        self.fit(llm_features, acoustic_features, labels)
        return self.transform(llm_features, acoustic_features)


def create_feature_fusion(
    config: dict,
    use_custom_acoustic: bool = False
) -> FeatureFusion:
    """Create feature fusion from config."""
    fs_config = config.get('feature_selection', {})
    
    if use_custom_acoustic:
        return CustomAcousticFusion(
            importance_threshold=fs_config.get('importance_threshold', 0.95),
            normalize=fs_config.get('normalize', True)
        )
    else:
        return FeatureFusion(
            importance_threshold=fs_config.get('importance_threshold', 0.95),
            normalize=fs_config.get('normalize', True)
        )


if __name__ == "__main__":
    # Test feature fusion
    np.random.seed(42)
    
    # Simulate features
    n_samples = 100
    llm_features = np.random.randn(n_samples, 4096)
    acoustic_features = np.random.randn(n_samples, 64)
    labels = np.random.randint(0, 2, n_samples)
    
    # Test fusion
    fusion = FeatureFusion(importance_threshold=0.95)
    selected = fusion.fit_transform(llm_features, acoustic_features, labels)
    
    print(f"\nSelected features shape: {selected.shape}")
    print(f"\nImportance breakdown:")
    for k, v in fusion.get_feature_importance_breakdown().items():
        print(f"  {k}: {v}")

