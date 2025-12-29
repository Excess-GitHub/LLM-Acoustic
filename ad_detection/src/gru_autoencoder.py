"""
GRU-based Autoencoder for generating fixed-dimension acoustic features.

Based on the paper's Section III-C: A supervised autoencoder with GRU architecture
is used to encode variable-length VGGish features into fixed-dimension vectors.

The autoencoder is trained with two losses:
1. Reconstruction loss: How well the decoder reconstructs the VGGish features
2. Classification loss: How well the encoded features predict AD vs HC
"""

import os
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class VGGishDataset(Dataset):
    """Dataset for VGGish sequences with labels."""
    
    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        max_length: Optional[int] = None
    ):
        """
        Args:
            sequences: List of VGGish feature arrays (n_windows, 128).
            labels: List of labels (0=HC, 1=AD).
            max_length: Maximum sequence length (for padding).
        """
        self.sequences = sequences
        self.labels = labels
        
        # Determine max length
        if max_length is None:
            self.max_length = max(seq.shape[0] for seq in sequences)
        else:
            self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Pad or truncate to max_length
        seq_len = seq.shape[0]
        if seq_len < self.max_length:
            # Pad with zeros
            padding = np.zeros((self.max_length - seq_len, seq.shape[1]))
            seq = np.vstack([seq, padding])
        elif seq_len > self.max_length:
            # Truncate
            seq = seq[:self.max_length]
        
        return {
            'sequence': torch.FloatTensor(seq),
            'label': torch.LongTensor([label]),
            'length': torch.LongTensor([min(seq_len, self.max_length)])
        }


class GRUEncoder(nn.Module):
    """GRU encoder for VGGish sequences."""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequences.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim).
            lengths: Sequence lengths (batch,).
            
        Returns:
            output: All hidden states (batch, seq_len, hidden_dim * num_directions).
            hidden: Final hidden state (num_layers * num_directions, batch, hidden_dim).
        """
        if lengths is not None:
            # Pack padded sequences for efficient processing
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, hidden = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, hidden = self.gru(x)
        
        return output, hidden


class GRUDecoder(nn.Module):
    """GRU decoder for reconstructing VGGish sequences."""
    
    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.gru = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        hidden: torch.Tensor,
        target_length: int,
        teacher_forcing_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode hidden state to sequence.
        
        Args:
            hidden: Initial hidden state from encoder.
            target_length: Length of sequence to generate.
            teacher_forcing_input: Optional input for teacher forcing.
            
        Returns:
            Reconstructed sequence (batch, target_length, output_dim).
        """
        batch_size = hidden.shape[1]
        device = hidden.device
        
        # Initialize input (start token - zeros)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=device)
        
        outputs = []
        h = hidden
        
        for t in range(target_length):
            output, h = self.gru(decoder_input, h)
            output = self.fc(output)
            outputs.append(output)
            
            if teacher_forcing_input is not None:
                decoder_input = teacher_forcing_input[:, t:t+1, :]
            else:
                decoder_input = output
        
        return torch.cat(outputs, dim=1)


class GRUAutoencoder(nn.Module):
    """
    Supervised GRU Autoencoder for acoustic feature encoding.
    
    Combines:
    1. Encoder: VGGish sequence -> fixed-dim hidden state
    2. Decoder: Hidden state -> reconstructed VGGish sequence
    3. Classifier: Hidden state -> AD/HC prediction
    
    From paper Section III-C: The hidden state from the GRU encoder,
    a fixed-size vector, is used as the acoustic feature vector.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.feature_dim = hidden_dim * self.num_directions * num_layers
        
        # Encoder
        self.encoder = GRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Decoder (always unidirectional for generation)
        self.decoder = GRUDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # Binary classification (AD vs HC)
        )
        
        # For bidirectional encoder, we need to project hidden state for decoder
        if bidirectional:
            self.hidden_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.hidden_proj = None
    
    def encode(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode VGGish sequences to fixed-dim vectors.
        
        Args:
            x: Input sequences (batch, seq_len, 128).
            lengths: Sequence lengths.
            
        Returns:
            Acoustic feature vectors (batch, feature_dim).
        """
        _, hidden = self.encoder(x, lengths)
        
        # Reshape hidden: (num_layers * num_dir, batch, hidden) -> (batch, feature_dim)
        batch_size = hidden.shape[1]
        feature_vec = hidden.permute(1, 0, 2).reshape(batch_size, -1)
        
        return feature_vec
    
    def decode(
        self,
        hidden: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """
        Decode hidden state to VGGish sequence.
        
        Args:
            hidden: Hidden state from encoder.
            target_length: Length of sequence to generate.
            
        Returns:
            Reconstructed sequence.
        """
        # Reshape feature_vec back to hidden state format
        batch_size = hidden.shape[0]
        num_layers = self.encoder.num_layers
        
        if self.encoder.bidirectional:
            # For bidirectional, we need to handle the projection
            hidden = hidden.view(batch_size, num_layers, 2, self.hidden_dim)
            # Take only forward direction for decoder
            hidden = hidden[:, :, 0, :]
            hidden = hidden.permute(1, 0, 2)  # (num_layers, batch, hidden)
        else:
            hidden = hidden.view(batch_size, num_layers, self.hidden_dim)
            hidden = hidden.permute(1, 0, 2)
        
        return self.decoder(hidden, target_length)
    
    def classify(self, feature_vec: torch.Tensor) -> torch.Tensor:
        """Classify from feature vector."""
        return self.classifier(feature_vec)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: Input VGGish sequences (batch, seq_len, 128).
            lengths: Sequence lengths.
            
        Returns:
            feature_vec: Encoded features (batch, feature_dim).
            reconstructed: Reconstructed sequences (batch, seq_len, 128).
            logits: Classification logits (batch, 2).
        """
        # Encode
        feature_vec = self.encode(x, lengths)
        
        # Decode
        target_length = x.shape[1]
        reconstructed = self.decode(feature_vec, target_length)
        
        # Classify
        logits = self.classify(feature_vec)
        
        return feature_vec, reconstructed, logits


class GRUAutoencoderTrainer:
    """
    Trainer for GRU Autoencoder.
    
    Trains with combined reconstruction and classification loss.
    Loss is normalized so both components are on similar scales.
    """
    
    def __init__(
        self,
        model: GRUAutoencoder,
        reconstruction_weight: float = 0.3,  # Lower weight for reconstruction
        learning_rate: float = 0.001,
        device: str = 'cuda',
        normalize_loss: bool = True  # Normalize losses to similar scale
    ):
        self.model = model.to(device)
        self.device = device
        self.reconstruction_weight = reconstruction_weight
        self.normalize_loss = normalize_loss
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.reconstruction_criterion = nn.MSELoss()
        self.classification_criterion = nn.CrossEntropyLoss()
        
        # Running stats for loss normalization
        self.recon_loss_avg = 1.0
        self.class_loss_avg = 1.0
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_class_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            sequences = batch['sequence'].to(self.device)
            labels = batch['label'].squeeze(-1).to(self.device)
            lengths = batch['length'].squeeze(-1)
            
            # Forward pass
            self.optimizer.zero_grad()
            feature_vec, reconstructed, logits = self.model(sequences, lengths)
            
            # Compute losses
            recon_loss = self.reconstruction_criterion(reconstructed, sequences)
            class_loss = self.classification_criterion(logits, labels)
            
            # Normalize losses to similar scale
            if self.normalize_loss:
                # Update running averages
                self.recon_loss_avg = 0.9 * self.recon_loss_avg + 0.1 * recon_loss.item()
                self.class_loss_avg = 0.9 * self.class_loss_avg + 0.1 * class_loss.item()
                
                # Normalize
                recon_loss_norm = recon_loss / (self.recon_loss_avg + 1e-8)
                class_loss_norm = class_loss / (self.class_loss_avg + 1e-8)
                
                # Combined loss (normalized)
                loss = (self.reconstruction_weight * recon_loss_norm + 
                       (1 - self.reconstruction_weight) * class_loss_norm)
            else:
                # Combined loss (raw)
                loss = (self.reconstruction_weight * recon_loss + 
                       (1 - self.reconstruction_weight) * class_loss)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics (use raw values for logging)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'class_loss': total_class_loss / n_batches,
            'accuracy': correct / total
        }
    
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_class_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].squeeze(-1).to(self.device)
                lengths = batch['length'].squeeze(-1)
                
                feature_vec, reconstructed, logits = self.model(sequences, lengths)
                
                recon_loss = self.reconstruction_criterion(reconstructed, sequences)
                class_loss = self.classification_criterion(logits, labels)
                loss = (self.reconstruction_weight * recon_loss + 
                       (1 - self.reconstruction_weight) * class_loss)
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_class_loss += class_loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'class_loss': total_class_loss / n_batches,
            'accuracy': correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        patience: int = 10,
        min_delta: float = 0.001,
        auto_split: bool = True
    ) -> dict:
        """
        Train the autoencoder with early stopping.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader (if None, uses train loss for early stopping).
            epochs: Number of epochs.
            patience: Early stopping patience (epochs without improvement).
            min_delta: Minimum change to qualify as improvement.
            auto_split: If True and no val_loader, track training loss for early stopping.
            
        Returns:
            Training history.
        """
        history = {'train': [], 'val': []}
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        best_epoch = 0
        
        # Determine what metric to use for early stopping
        use_val = val_loader is not None
        metric_name = "Val" if use_val else "Train"
        
        print(f"GRU-AE Training: {epochs} epochs, patience={patience}, early_stop_on={metric_name}")
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            history['train'].append(train_metrics)
            
            log_str = f"Epoch {epoch+1}/{epochs} - "
            log_str += f"Loss: {train_metrics['loss']:.4f} "
            log_str += f"(R:{train_metrics['recon_loss']:.1f}, C:{train_metrics['class_loss']:.3f}), "
            log_str += f"Acc: {train_metrics['accuracy']:.4f}"
            
            # Get metric for early stopping
            if use_val:
                val_metrics = self.evaluate(val_loader)
                history['val'].append(val_metrics)
                log_str += f" | Val Loss: {val_metrics['loss']:.4f}, "
                log_str += f"Acc: {val_metrics['accuracy']:.4f}"
                current_loss = val_metrics['loss']
            else:
                # Use training loss for early stopping
                current_loss = train_metrics['loss']
            
            # Early stopping check
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch + 1
                log_str += " *"  # Mark best epoch
            else:
                patience_counter += 1
                log_str += f" (patience: {patience_counter}/{patience})"
            
            print(log_str)
            
            # Check early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best was epoch {best_epoch})")
                break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Loaded best model from epoch {best_epoch} (loss: {best_loss:.4f})")
        
        return history
    
    def extract_features(
        self,
        sequences: List[np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract acoustic feature vectors from VGGish sequences.
        
        Args:
            sequences: List of VGGish feature arrays.
            batch_size: Batch size for processing.
            
        Returns:
            Feature vectors (n_samples, feature_dim).
        """
        self.model.eval()
        
        # Create dataset (dummy labels)
        dataset = VGGishDataset(sequences, [0] * len(sequences))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        features = []
        with torch.no_grad():
            for batch in loader:
                seq = batch['sequence'].to(self.device)
                lengths = batch['length'].squeeze(-1)
                
                feature_vec = self.model.encode(seq, lengths)
                features.append(feature_vec.cpu().numpy())
        
        return np.vstack(features)


def create_gru_autoencoder(config: dict) -> GRUAutoencoder:
    """Create GRU autoencoder from config."""
    gru_config = config.get('gru_autoencoder', {})
    vggish_config = config.get('vggish', {})
    
    return GRUAutoencoder(
        input_dim=vggish_config.get('embedding_dim', 128),
        hidden_dim=gru_config.get('hidden_dim', 64),
        num_layers=gru_config.get('num_layers', 1),
        dropout=gru_config.get('dropout', 0.1),
        bidirectional=gru_config.get('bidirectional', False)
    )


if __name__ == "__main__":
    # Test GRU autoencoder
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_gru_autoencoder(config)
    print(f"Model created with feature dimension: {model.feature_dim}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 128)
    lengths = torch.randint(50, seq_len, (batch_size,))
    
    feature_vec, reconstructed, logits = model(x, lengths)
    
    print(f"Input shape: {x.shape}")
    print(f"Feature vector shape: {feature_vec.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Logits shape: {logits.shape}")

