"""
Pitt Corpus metadata parsing utilities.

Handles:
- Parsing CHAT (.cha) transcript files
- Extracting participant demographics (age, gender, diagnosis, MMSE)
- Extracting speaker timestamps from transcripts
- Loading entire Pitt Corpus structure
"""

import os
import re
import logging
from typing import Dict, List, Tuple, Optional, Generator, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParticipantInfo:
    """Container for participant metadata from CHAT files."""
    participant_id: str
    session: str
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None
    mmse: Optional[int] = None
    group: str = "Unknown"  # "AD" or "HC"
    file_path: Optional[str] = None
    audio_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'participant_id': self.participant_id,
            'session': self.session,
            'age': self.age,
            'gender': self.gender,
            'diagnosis': self.diagnosis,
            'mmse': self.mmse,
            'group': self.group,
            'file_path': self.file_path,
            'audio_path': self.audio_path
        }


@dataclass 
class SpeakerTurn:
    """Container for a single speaker turn with timestamps."""
    speaker: str  # "PAR" (participant) or "INV" (investigator)
    text: str
    start_time: float  # in seconds
    end_time: float  # in seconds
    word_timestamps: List[Tuple[str, float, float]] = field(default_factory=list)


def parse_chat_file(chat_path: str) -> Tuple[ParticipantInfo, List[SpeakerTurn]]:
    """
    Parse a CHAT transcript file to extract participant info and speaker turns.
    
    CHAT format reference:
    - @ID line contains demographics: eng|Pitt|PAR|age;|gender|diagnosis||Participant|mmse||
    - *PAR: participant speech with timestamps (e.g., "text here . 1234_5678")
    - *INV: investigator speech with timestamps
    - %wor: word-level timestamps
    
    Parameters
    ----------
    chat_path : str
        Path to .cha file
        
    Returns
    -------
    participant_info : ParticipantInfo
        Parsed participant demographics
    speaker_turns : List[SpeakerTurn]
        List of speaker turns with timestamps
    """
    if not os.path.exists(chat_path):
        raise FileNotFoundError(f"CHAT file not found: {chat_path}")
    
    # Extract participant ID and session from filename
    filename = os.path.basename(chat_path)
    file_id = os.path.splitext(filename)[0]  # e.g., "001-0"
    parts = file_id.split('-')
    participant_id = parts[0] if parts else file_id
    session = parts[1] if len(parts) > 1 else "0"
    
    participant_info = ParticipantInfo(
        participant_id=participant_id,
        session=session,
        file_path=chat_path
    )
    speaker_turns = []
    
    with open(chat_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_turn = None
    word_line_buffer = None
    
    for line in lines:
        line = line.strip()
        
        # Parse @ID line for participant info
        if line.startswith('@ID:') and 'PAR' in line:
            participant_info = _parse_id_line(line, participant_info)
        
        # Parse @ID line to determine group from path
        # (Dementia vs Control determined by folder structure)
        
        # Parse speaker turns (*PAR: or *INV:)
        if line.startswith('*PAR:') or line.startswith('*INV:'):
            # Save previous turn if exists
            if current_turn is not None:
                speaker_turns.append(current_turn)
            
            current_turn = _parse_speaker_line(line)
            word_line_buffer = None
            
        # Parse word-level timestamps (%wor:)
        elif line.startswith('%wor:') and current_turn is not None:
            word_timestamps = _parse_word_timestamps(line)
            current_turn.word_timestamps = word_timestamps
    
    # Don't forget the last turn
    if current_turn is not None:
        speaker_turns.append(current_turn)
    
    return participant_info, speaker_turns


def _parse_id_line(line: str, info: ParticipantInfo) -> ParticipantInfo:
    """
    Parse @ID line to extract participant demographics.
    
    Format: eng|Pitt|PAR|age;|gender|diagnosis||Participant|mmse||
    Example: eng|Pitt|PAR|57;|male|ProbableAD||Participant|18||
    """
    try:
        # Remove @ID: prefix
        content = line.replace('@ID:', '').strip()
        parts = content.split('|')
        
        if len(parts) >= 9:
            # Extract age (format: "57;" or just "57")
            age_str = parts[3].replace(';', '').strip()
            if age_str.isdigit():
                info.age = int(age_str)
            
            # Extract gender
            gender = parts[4].strip().lower()
            if gender in ['male', 'female', 'm', 'f']:
                info.gender = 'M' if gender in ['male', 'm'] else 'F'
            
            # Extract diagnosis
            diagnosis = parts[5].strip()
            if diagnosis:
                info.diagnosis = diagnosis
                # Set group based on diagnosis
                if 'AD' in diagnosis.upper() or 'DEMENTIA' in diagnosis.upper() or 'PROBABLE' in diagnosis.upper():
                    info.group = 'AD'
                elif 'CONTROL' in diagnosis.upper():
                    info.group = 'HC'
            
            # Extract MMSE score (typically in position 8)
            mmse_str = parts[8].strip()
            if mmse_str.isdigit():
                info.mmse = int(mmse_str)
                
    except Exception as e:
        logger.warning(f"Error parsing @ID line: {line}, error: {e}")
    
    return info


def _parse_speaker_line(line: str) -> SpeakerTurn:
    """
    Parse a speaker line (*PAR: or *INV:) with timestamps.
    
    Format: *PAR: text here . 1234_5678
    Timestamps are in milliseconds.
    """
    # Determine speaker
    if line.startswith('*PAR:'):
        speaker = 'PAR'
        content = line[5:].strip()
    elif line.startswith('*INV:'):
        speaker = 'INV'
        content = line[5:].strip()
    else:
        speaker = 'UNK'
        content = line.strip()
    
    # Extract timestamp from end of line (format: 1234_5678)
    timestamp_pattern = r'(\d+)_(\d+)\s*$'
    match = re.search(timestamp_pattern, content)
    
    start_time = 0.0
    end_time = 0.0
    text = content
    
    if match:
        start_ms = int(match.group(1))
        end_ms = int(match.group(2))
        start_time = start_ms / 1000.0  # Convert to seconds
        end_time = end_ms / 1000.0
        # Remove timestamp from text
        text = content[:match.start()].strip()
    
    return SpeakerTurn(
        speaker=speaker,
        text=text,
        start_time=start_time,
        end_time=end_time
    )


def _parse_word_timestamps(line: str) -> List[Tuple[str, float, float]]:
    """
    Parse word-level timestamps from %wor: line.
    
    Format: %wor: word1 1234_5678 word2 2345_6789 ...
    """
    content = line.replace('%wor:', '').strip()
    word_timestamps = []
    
    # Pattern: word followed by timestamp
    pattern = r'(\S+)\s+(\d+)_(\d+)'
    matches = re.findall(pattern, content)
    
    for word, start_ms, end_ms in matches:
        start_time = int(start_ms) / 1000.0
        end_time = int(end_ms) / 1000.0
        word_timestamps.append((word, start_time, end_time))
    
    return word_timestamps


def extract_speaker_timestamps(
    speaker_turns: List[SpeakerTurn],
    speaker: str = 'PAR'
) -> List[Tuple[float, float]]:
    """
    Extract timestamps for a specific speaker.
    
    Parameters
    ----------
    speaker_turns : List[SpeakerTurn]
        List of speaker turns from parse_chat_file
    speaker : str
        Speaker code ('PAR' for participant, 'INV' for investigator)
        
    Returns
    -------
    List[Tuple[float, float]]
        List of (start_time, end_time) tuples in seconds
    """
    timestamps = []
    for turn in speaker_turns:
        if turn.speaker == speaker and turn.end_time > turn.start_time:
            timestamps.append((turn.start_time, turn.end_time))
    return timestamps


def get_participant_info(
    chat_path: str,
    group_from_path: bool = True
) -> ParticipantInfo:
    """
    Quick extraction of participant info without full parsing.
    
    Parameters
    ----------
    chat_path : str
        Path to .cha file
    group_from_path : bool
        Whether to infer group from file path (Dementia/Control folders)
        
    Returns
    -------
    ParticipantInfo
        Participant demographics
    """
    info, _ = parse_chat_file(chat_path)
    
    # Infer group from path if not set
    if group_from_path and info.group == "Unknown":
        path_lower = chat_path.lower()
        if 'dementia' in path_lower:
            info.group = 'AD'
        elif 'control' in path_lower:
            info.group = 'HC'
    
    return info


def load_pitt_corpus(
    root_path: str,
    task: str = 'cookie',
    audio_format: str = 'WAV',
    include_metadata: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """
    Load Pitt Corpus data, yielding audio paths with metadata.
    
    Parameters
    ----------
    root_path : str
        Path to Pitt Corpus root (containing Media/ and Transcripts/ folders)
    task : str
        Task to load ('cookie', 'fluency', 'recall', 'sentence')
    audio_format : str
        Audio format to use ('WAV' or 'MP3')
    include_metadata : bool
        Whether to parse and include metadata from transcripts
        
    Yields
    ------
    dict
        Dictionary with keys: 'audio_path', 'transcript_path', 'participant_info'
    """
    media_dir = os.path.join(root_path, 'Media')
    transcripts_dir = os.path.join(root_path, 'Transcripts', 'Pitt')
    
    # Process both groups
    for group_folder, group_label in [('Dementia', 'AD'), ('Control', 'HC')]:
        # Construct paths
        task_cap = task.capitalize() if task != 'cookie' else 'Cookie'
        audio_dir = os.path.join(media_dir, group_folder, task_cap, audio_format)
        transcript_dir = os.path.join(transcripts_dir, group_folder, task.lower())
        
        if not os.path.exists(audio_dir):
            logger.warning(f"Audio directory not found: {audio_dir}")
            continue
        
        # Get all audio files
        audio_ext = '.wav' if audio_format == 'WAV' else '.mp3'
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(audio_ext)]
        
        for audio_file in sorted(audio_files):
            audio_path = os.path.join(audio_dir, audio_file)
            
            # Find corresponding transcript
            transcript_name = os.path.splitext(audio_file)[0] + '.cha'
            transcript_path = os.path.join(transcript_dir, transcript_name)
            
            # Parse metadata if requested and transcript exists
            participant_info = None
            speaker_turns = None
            
            if include_metadata and os.path.exists(transcript_path):
                try:
                    participant_info, speaker_turns = parse_chat_file(transcript_path)
                    participant_info.group = group_label
                    participant_info.audio_path = audio_path
                except Exception as e:
                    logger.warning(f"Error parsing transcript {transcript_path}: {e}")
            else:
                # Create basic info from filename
                file_id = os.path.splitext(audio_file)[0]
                parts = file_id.split('-')
                participant_info = ParticipantInfo(
                    participant_id=parts[0] if parts else file_id,
                    session=parts[1] if len(parts) > 1 else "0",
                    group=group_label,
                    audio_path=audio_path
                )
            
            yield {
                'audio_path': audio_path,
                'transcript_path': transcript_path if os.path.exists(transcript_path) else None,
                'participant_info': participant_info,
                'speaker_turns': speaker_turns,
                'group': group_label,
                'task': task
            }


def get_corpus_stats(root_path: str, task: str = 'cookie') -> Dict[str, Any]:
    """
    Get statistics about the corpus.
    
    Parameters
    ----------
    root_path : str
        Path to Pitt Corpus root
    task : str
        Task to analyze
        
    Returns
    -------
    dict
        Corpus statistics
    """
    stats = {
        'total_files': 0,
        'dementia_files': 0,
        'control_files': 0,
        'unique_participants_dementia': set(),
        'unique_participants_control': set(),
        'files_with_metadata': 0,
        'ages': [],
        'mmse_scores': []
    }
    
    for item in load_pitt_corpus(root_path, task=task):
        stats['total_files'] += 1
        
        if item['group'] == 'AD':
            stats['dementia_files'] += 1
            stats['unique_participants_dementia'].add(item['participant_info'].participant_id)
        else:
            stats['control_files'] += 1
            stats['unique_participants_control'].add(item['participant_info'].participant_id)
        
        if item['participant_info'].age is not None:
            stats['files_with_metadata'] += 1
            stats['ages'].append(item['participant_info'].age)
        
        if item['participant_info'].mmse is not None:
            stats['mmse_scores'].append(item['participant_info'].mmse)
    
    # Convert sets to counts
    stats['unique_participants_dementia'] = len(stats['unique_participants_dementia'])
    stats['unique_participants_control'] = len(stats['unique_participants_control'])
    
    return stats

