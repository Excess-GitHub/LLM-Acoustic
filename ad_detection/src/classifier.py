"""
Classification models for AD detection.

The paper compares multiple classifiers:
- SVC (Support Vector Classifier) - BEST
- Logistic Regression (LR)
- Random Forest (RF)
- Gradient Boosting (GB)
- XGBoost (XGB)
- Artificial Neural Network (ANN)
- Stacking Classifier (SC)

The best results are achieved with SVC (RBF kernel) as shown in Table III.
"""

import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ANNClassifier(nn.Module):
    """
    Artificial Neural Network classifier.
    
    From paper Section III-E:
    - 3 fully connected layers
    - First layer: 4096 -> 1024 (for LLM features)
    - Second layer: 1024 -> 128
    - Output layer: 128 -> 1
    - ReLU activations, batch norm, dropout
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = [1024, 128],
        dropout: float = 0.01
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ANNClassifierWrapper:
    """
    Sklearn-compatible wrapper for ANN classifier.
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = [1024, 128],
        dropout: float = 0.01,
        learning_rate: float = 1e-5,
        epochs: int = 100,
        batch_size: int = 16,
        patience: int = 10,
        device: str = 'cuda'
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ANNClassifierWrapper':
        """Fit the ANN classifier."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model
        self.model = ANNClassifier(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            
            # Early stopping
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = torch.sigmoid(self.model(X_tensor))
            predictions = (outputs > 0.5).cpu().numpy().flatten()
        
        return predictions.astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = torch.sigmoid(self.model(X_tensor)).cpu().numpy()
            proba = np.hstack([1 - outputs, outputs])
        
        return proba


class ADClassifier:
    """
    Main classifier class for AD detection.
    
    Supports multiple classifier types as compared in the paper.
    """
    
    SUPPORTED_CLASSIFIERS = ['svc', 'lr', 'rf', 'gb', 'xgb', 'ann', 'stacking']
    
    def __init__(
        self,
        classifier_type: str = 'svc',
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize classifier.
        
        Args:
            classifier_type: Type of classifier ('svc', 'lr', 'rf', 'gb', 'xgb', 'ann', 'stacking').
            random_state: Random state for reproducibility.
            **kwargs: Classifier-specific parameters.
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = None
        self._create_model()
    
    def _create_model(self):
        """Create the classifier model."""
        if self.classifier_type == 'svc':
            # SVC with RBF kernel (best from paper)
            self.model = SVC(
                kernel=self.kwargs.get('kernel', 'rbf'),
                C=self.kwargs.get('C', 1.0),
                gamma=self.kwargs.get('gamma', 'scale'),
                probability=True,
                random_state=self.random_state
            )
        
        elif self.classifier_type == 'lr':
            # Logistic Regression
            self.model = LogisticRegression(
                max_iter=self.kwargs.get('max_iter', 1000),
                C=self.kwargs.get('C', 1.0),
                penalty=self.kwargs.get('penalty', 'l2'),
                random_state=self.random_state
            )
        
        elif self.classifier_type == 'rf':
            # Random Forest
            self.model = RandomForestClassifier(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', None),
                random_state=self.random_state
            )
        
        elif self.classifier_type == 'gb':
            # Gradient Boosting
            self.model = GradientBoostingClassifier(
                n_estimators=self.kwargs.get('n_estimators', 50),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                random_state=self.random_state
            )
        
        elif self.classifier_type == 'xgb':
            # XGBoost
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    n_estimators=self.kwargs.get('n_estimators', 10),
                    eval_metric='logloss',
                    random_state=self.random_state,
                    use_label_encoder=False
                )
            except ImportError:
                print("XGBoost not installed, falling back to GradientBoosting")
                self.model = GradientBoostingClassifier(
                    n_estimators=50,
                    random_state=self.random_state
                )
        
        elif self.classifier_type == 'ann':
            # Artificial Neural Network
            self.model = ANNClassifierWrapper(
                input_dim=self.kwargs.get('input_dim', 4096),
                hidden_dims=self.kwargs.get('hidden_dims', [1024, 128]),
                dropout=self.kwargs.get('dropout', 0.01),
                learning_rate=self.kwargs.get('learning_rate', 1e-5),
                device=self.kwargs.get('device', 'cuda')
            )
        
        elif self.classifier_type == 'stacking':
            # Stacking Classifier (ensemble)
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)),
                ('svc', SVC(kernel='rbf', probability=True, random_state=self.random_state))
            ]
            
            try:
                from xgboost import XGBClassifier
                estimators.append(('xgb', XGBClassifier(
                    n_estimators=50,
                    eval_metric='logloss',
                    random_state=self.random_state,
                    use_label_encoder=False
                )))
            except ImportError:
                pass
            
            self.model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5
            )
        
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ADClassifier':
        """Fit the classifier."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    ) -> Dict[str, float]:
        """
        Evaluate the classifier.
        
        Args:
            X: Feature array.
            y: True labels.
            metrics: Metrics to compute.
            
        Returns:
            Dictionary of metric scores.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        results = {}
        
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y, y_pred)
        
        if 'precision' in metrics:
            results['precision'] = precision_score(y, y_pred, zero_division=0)
        
        if 'recall' in metrics:
            results['recall'] = recall_score(y, y_pred, zero_division=0)
        
        if 'f1' in metrics:
            results['f1'] = f1_score(y, y_pred, zero_division=0)
        
        if 'roc_auc' in metrics and y_proba is not None:
            try:
                results['roc_auc'] = roc_auc_score(y, y_proba)
            except ValueError:
                results['roc_auc'] = 0.5
        
        return results


def create_classifier(config: dict) -> ADClassifier:
    """Create classifier from config."""
    clf_config = config.get('classifier', {})
    clf_type = clf_config.get('type', 'svc')
    
    # Get classifier-specific parameters
    clf_params = clf_config.get(clf_type, {})
    
    return ADClassifier(
        classifier_type=clf_type,
        **clf_params
    )


def evaluate_with_repeats(
    classifier_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_repeats: int = 25,
    **kwargs
) -> Dict[str, Tuple[float, float]]:
    """
    Evaluate classifier with multiple random initializations.
    
    From paper: "metrics were computed 25 times, each initialization 
    using a different random state, with the results subsequently averaged"
    
    Args:
        classifier_type: Type of classifier.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        n_repeats: Number of repetitions.
        **kwargs: Classifier parameters.
        
    Returns:
        Dictionary with metric mean and std.
    """
    all_results = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
    
    for seed in range(n_repeats):
        clf = ADClassifier(
            classifier_type=classifier_type,
            random_state=seed,
            **kwargs
        )
        clf.fit(X_train, y_train)
        results = clf.evaluate(X_test, y_test)
        
        for metric, value in results.items():
            all_results[metric].append(value)
    
    # Compute mean and std
    summary = {}
    for metric, values in all_results.items():
        summary[metric] = (np.mean(values), np.std(values))
    
    return summary


if __name__ == "__main__":
    # Test classifiers
    np.random.seed(42)
    
    # Generate dummy data
    n_train, n_test = 100, 30
    n_features = 100
    
    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.randint(0, 2, n_train)
    X_test = np.random.randn(n_test, n_features)
    y_test = np.random.randint(0, 2, n_test)
    
    # Test each classifier type
    for clf_type in ['svc', 'lr', 'rf', 'gb', 'ann']:
        print(f"\nTesting {clf_type.upper()}:")
        clf = ADClassifier(classifier_type=clf_type)
        clf.fit(X_train, y_train)
        results = clf.evaluate(X_test, y_test)
        
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

