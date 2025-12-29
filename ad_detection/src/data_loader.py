"""
Data loader for Pitt Corpus dataset.

Handles loading audio files, transcripts, and metadata from the Pitt Corpus
for the Cookie Theft picture description task.
"""

import os
import re
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np
from tqdm import tqdm


@dataclass
class PittSample:
    """Single sample from Pitt Corpus."""
    participant_id: str
    session: int
    audio_path: str
    transcript_path: str
    group: str  # 'AD' or 'HC'
    age: Optional[float] = None
    gender: Optional[str] = None
    mmse: Optional[float] = None
    transcript_text: Optional[str] = None
    

def parse_cha_transcript(cha_path: str) -> Dict[str, Any]:
    """
    Parse CHAT (.cha) transcript file.
    
    Returns:
        Dictionary with metadata and participant utterances.
    """
    metadata = {
        'age': None,
        'gender': None,
        'mmse': None,
        'utterances': [],
        'participant_text': ''
    }
    
    try:
        with open(cha_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(cha_path, 'r', encoding='latin-1') as f:
            content = f.read()
    
    lines = content.split('\n')
    
    for line in lines:
        # Parse participant ID line for demographics
        # Format: @ID: eng|Pitt|PAR|57;|male|ProbableAD||Participant|18||
        if line.startswith('@ID:') and 'PAR' in line:
            parts = line.split('|')
            if len(parts) >= 9:
                # Age is in format "57;" 
                age_str = parts[3].replace(';', '').strip()
                if age_str:
                    try:
                        metadata['age'] = float(age_str)
                    except ValueError:
                        pass
                
                # Gender
                gender_str = parts[4].strip().lower()
                if gender_str in ['male', 'm']:
                    metadata['gender'] = 'M'
                elif gender_str in ['female', 'f']:
                    metadata['gender'] = 'F'
                
                # MMSE score is typically in position 8
                mmse_str = parts[8].strip() if len(parts) > 8 else ''
                if mmse_str:
                    try:
                        metadata['mmse'] = float(mmse_str)
                    except ValueError:
                        pass
        
        # Parse participant utterances (*PAR: lines)
        if line.startswith('*PAR:'):
            # Remove timing information (numbers like 1234_5678)
            utterance = line[5:].strip()
            utterance = re.sub(r'\d+_\d+', '', utterance)
            # Remove CHAT coding markers
            utterance = re.sub(r'\[.*?\]', '', utterance)  # Remove [+ exc], [*] etc
            utterance = re.sub(r'&\S+', '', utterance)  # Remove &-um, &+flow etc
            utterance = re.sub(r'<.*?>', '', utterance)  # Remove <...>
            utterance = re.sub(r'\(\.\)', '', utterance)  # Remove (.)
            utterance = re.sub(r'\s+', ' ', utterance).strip()  # Clean whitespace
            
            if utterance and utterance != '.':
                metadata['utterances'].append(utterance)
    
    # Combine all participant utterances
    metadata['participant_text'] = ' '.join(metadata['utterances'])
    
    return metadata


def load_pitt_corpus(
    corpus_root: str,
    task: str = 'cookie',
    groups: List[str] = ['Dementia', 'Control']
) -> List[PittSample]:
    """
    Load Pitt Corpus dataset.
    
    Args:
        corpus_root: Path to Pitt Corpus root directory.
        task: Task name ('cookie', 'fluency', 'recall', 'sentence').
        groups: Groups to load ('Dementia', 'Control').
        
    Returns:
        List of PittSample objects.
    """
    samples = []
    corpus_root = Path(corpus_root)
    
    for group in groups:
        # Determine label
        label = 'AD' if group == 'Dementia' else 'HC'
        
        # Find audio files
        audio_dir = corpus_root / 'Media' / group / task.capitalize() / 'WAV'
        if not audio_dir.exists():
            print(f"Warning: Audio directory not found: {audio_dir}")
            continue
            
        # Find transcript files  
        transcript_dir = corpus_root / 'Transcripts' / 'Pitt' / group / task.lower()
        if not transcript_dir.exists():
            print(f"Warning: Transcript directory not found: {transcript_dir}")
            continue
        
        # Get all audio files
        audio_files = sorted(glob.glob(str(audio_dir / '*.wav')))
        
        for audio_path in audio_files:
            audio_name = os.path.basename(audio_path)
            base_name = audio_name.replace('.wav', '')
            
            # Parse participant ID and session from filename (e.g., "001-0" -> id=001, session=0)
            parts = base_name.split('-')
            if len(parts) >= 2:
                participant_id = parts[0]
                try:
                    session = int(parts[1])
                except ValueError:
                    session = 0
            else:
                participant_id = base_name
                session = 0
            
            # Find corresponding transcript
            transcript_path = transcript_dir / f"{base_name}.cha"
            if not transcript_path.exists():
                print(f"Warning: Transcript not found for {audio_path}")
                continue
            
            # Parse transcript for metadata
            transcript_data = parse_cha_transcript(str(transcript_path))
            
            sample = PittSample(
                participant_id=participant_id,
                session=session,
                audio_path=audio_path,
                transcript_path=str(transcript_path),
                group=label,
                age=transcript_data['age'],
                gender=transcript_data['gender'],
                mmse=transcript_data['mmse'],
                transcript_text=transcript_data['participant_text']
            )
            samples.append(sample)
    
    print(f"Loaded {len(samples)} samples from Pitt Corpus")
    print(f"  AD: {sum(1 for s in samples if s.group == 'AD')}")
    print(f"  HC: {sum(1 for s in samples if s.group == 'HC')}")
    
    return samples


class PittCorpusDataset:
    """
    Dataset class for Pitt Corpus with train/test split support.
    """
    
    def __init__(
        self,
        corpus_root: str,
        custom_features_csv: Optional[str] = None,
        task: str = 'cookie',
        random_seed: int = 42
    ):
        self.corpus_root = corpus_root
        self.custom_features_csv = custom_features_csv
        self.task = task
        self.random_seed = random_seed
        
        # Load samples
        self.samples = load_pitt_corpus(corpus_root, task)
        
        # Load custom audio features if provided
        self.custom_features_df = None
        if custom_features_csv and os.path.exists(custom_features_csv):
            self.custom_features_df = pd.read_csv(custom_features_csv)
            print(f"Loaded custom features: {self.custom_features_df.shape}")
    
    def get_sample_id(self, sample: PittSample) -> str:
        """Generate unique sample ID."""
        return f"{sample.participant_id}-{sample.session}"
    
    def get_custom_features(self, sample: PittSample) -> Optional[np.ndarray]:
        """Get custom audio features for a sample."""
        if self.custom_features_df is None:
            return None
        
        sample_id = self.get_sample_id(sample)
        pid, session = sample_id.split('-')
        
        # Match by participant_id and session
        mask = (
            (self.custom_features_df['participant_id'].astype(str).str.zfill(3) == pid.zfill(3)) &
            (self.custom_features_df['session'].astype(int) == int(session))
        )
        
        if mask.sum() == 0:
            return None
        
        row = self.custom_features_df[mask].iloc[0]
        
        # Get numeric feature columns (exclude metadata)
        exclude_cols = ['participant_id', 'session', 'group', 'age', 'mmse', 'gender']
        feature_cols = [c for c in self.custom_features_df.columns if c not in exclude_cols]
        
        features = row[feature_cols].values.astype(np.float32)
        return features
    
    def get_train_test_split(
        self,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[List[PittSample], List[PittSample]]:
        """
        Split dataset into train and test sets.
        
        Uses participant-level split to avoid data leakage.
        """
        from sklearn.model_selection import train_test_split
        
        # Group samples by participant
        participant_samples = {}
        for sample in self.samples:
            pid = sample.participant_id
            if pid not in participant_samples:
                participant_samples[pid] = []
            participant_samples[pid].append(sample)
        
        # Get unique participants with their labels
        participants = list(participant_samples.keys())
        labels = [participant_samples[p][0].group for p in participants]
        
        # Split at participant level
        if stratify:
            train_pids, test_pids = train_test_split(
                participants,
                test_size=test_size,
                random_state=self.random_seed,
                stratify=labels
            )
        else:
            train_pids, test_pids = train_test_split(
                participants,
                test_size=test_size,
                random_state=self.random_seed
            )
        
        # Collect samples
        train_samples = []
        test_samples = []
        
        for pid in train_pids:
            train_samples.extend(participant_samples[pid])
        for pid in test_pids:
            test_samples.extend(participant_samples[pid])
        
        return train_samples, test_samples
    
    def get_cv_folds(
        self,
        n_folds: int = 5
    ) -> List[Tuple[List[PittSample], List[PittSample]]]:
        """
        Generate cross-validation folds.
        
        Uses participant-level split.
        If n_folds=1, uses simple 80/20 train/test split.
        """
        # For n_folds=1, use simple train/test split
        if n_folds == 1:
            train_samples, test_samples = self.get_train_test_split(test_size=0.2)
            return [(train_samples, test_samples)]
        
        from sklearn.model_selection import StratifiedKFold
        
        # Group by participant
        participant_samples = {}
        for sample in self.samples:
            pid = sample.participant_id
            if pid not in participant_samples:
                participant_samples[pid] = []
            participant_samples[pid].append(sample)
        
        participants = list(participant_samples.keys())
        labels = [participant_samples[p][0].group for p in participants]
        
        # Create folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        folds = []
        
        for train_idx, test_idx in skf.split(participants, labels):
            train_pids = [participants[i] for i in train_idx]
            test_pids = [participants[i] for i in test_idx]
            
            train_samples = []
            test_samples = []
            
            for pid in train_pids:
                train_samples.extend(participant_samples[pid])
            for pid in test_pids:
                test_samples.extend(participant_samples[pid])
            
            folds.append((train_samples, test_samples))
        
        return folds
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert samples to pandas DataFrame."""
        records = []
        for sample in self.samples:
            records.append({
                'participant_id': sample.participant_id,
                'session': sample.session,
                'audio_path': sample.audio_path,
                'transcript_path': sample.transcript_path,
                'group': sample.group,
                'age': sample.age,
                'gender': sample.gender,
                'mmse': sample.mmse,
                'transcript': sample.transcript_text
            })
        return pd.DataFrame(records)


if __name__ == "__main__":
    # Test data loading
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = PittCorpusDataset(
        corpus_root=config['paths']['pitt_corpus_root'],
        custom_features_csv=config['paths']['custom_audio_features']
    )
    
    # Print statistics
    df = dataset.to_dataframe()
    print("\nDataset Statistics:")
    print(df.groupby('group').agg({
        'participant_id': 'nunique',
        'audio_path': 'count',
        'age': 'mean',
        'mmse': 'mean'
    }))
    
    # Test CV folds
    folds = dataset.get_cv_folds(n_folds=5)
    for i, (train, test) in enumerate(folds):
        print(f"\nFold {i+1}: Train={len(train)}, Test={len(test)}")

