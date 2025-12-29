# Audio Feature Extraction Pipeline for Pitt Corpus

Extracts clinician-interpretable acoustic features from the Pitt Corpus for Alzheimer's disease and mild cognitive impairment detection.

## Scientific Foundation

This pipeline implements features that have demonstrated clinical validity across multiple independent studies. The feature set draws from two methodologically complementary sources:

**Barragán Pulido et al. (2020)** reviewed 90+ studies of automatic speech analysis for AD detection, establishing that conventional prosodic and temporal features (pause duration, pitch variation, jitter, shimmer, harmonics-to-noise ratio) achieve accuracies between 70–90% when properly selected and applied to spontaneous speech. Critically, this review identified that combining complementary feature families—duration, prosody, and voice quality—improves discriminative performance while maintaining clinical interpretability.

**Pérez-Toro et al. (2022)** demonstrated that timing features (pause duration, speech segment statistics) extracted via energy-based Voice Activity Detection remain highly predictive on the Pitt Corpus. Using interpretable acoustic, phonemic, and duration features with ForestNet, they achieved 79% unweighted average recall (UAR) in discriminating AD from healthy controls, with duration-based features contributing most discriminative power. A subsequent analysis on the ADReSSo 2021 benchmark found that pause timing and voiced rates were among the top-ranked features for AD classification across multiple studies.

**Pérez-Toro et al. (2025)** evaluated pause-based timing markers and lexico-semantic features across English and Spanish speakers, demonstrating that pause duration and frequency generalize well across languages—a critical finding for clinical applicability. Within English, timing markers combined with lexico-semantic features achieved AUC 0.88, and pause-only features alone achieved AUC 0.75 cross-linguistically, supporting the hypothesis that temporal speech changes reflect common cognitive mechanisms in dementia.

## Features Extracted

### Timing Features (17 total)

Energy-based Voice Activity Detection separates speech from pause segments using a 25 ms frame window and 10 ms hop, following the cross-linguistic VAD implementation (Pérez-Toro et al., 2025). Functionals computed over pause and speech segment durations:

| Feature | Definition | AD Clinical Pattern |
|---------|-----------|-------------------|
| `pause_ratio` | Total pause time / recording duration | Elevated in AD (word retrieval difficulty) |
| `speech_ratio` | Total speech time / recording duration | Reduced in AD |
| `pause_dur_mean, std, min, max` | Pause duration statistics (seconds) | Longer pauses in AD (semantic access delay) |
| `speech_dur_mean, std, min, max` | Speech segment duration statistics | Shorter, fragmented speech in AD |
| `pause_dur_skew, kurt` | Shape of pause duration distribution | Increased skewness in AD (variable retrieval difficulty) |
| `speech_dur_skew, kurt` | Shape of speech segment distribution | - |
| `num_pauses_per_sec` | Pause frequency | Higher in AD |

### Prosodic Features (11 total)

Fundamental frequency (F₀) extracted using pYIN algorithm (Mauch & Dixon, 2014) across 25 ms frames. Statistics computed only on voiced frames. Intensity computed from energy contour.

| Feature | Definition | AD Clinical Pattern |
|---------|-----------|-------------------|
| `f0_mean, std` | F₀ center and variability (Hz) | Lower mean in elderly, reduced range in AD |
| `f0_min, max, range` | F₀ extremes | Reduced range → monotonous speech in AD |
| `intensity_mean, std` | Loudness (dB) | Flatter intensity contour in AD |
| `intensity_min, max, range` | Loudness extremes | - |
| `voice_rate` | Voiced frames / total frames | Lower in AD (reduced vocalization) |

### Voice Quality Features (8 total)

Periodic perturbation measures computed on pitch-extracted frames. Harmonics-to-noise ratio (HNR) reflects voice periodicity.

| Feature | Definition | AD Clinical Pattern |
|---------|-----------|-------------------|
| `jitter_mean, std` | F₀ cycle-to-cycle perturbation (%) | Elevated in AD/aging (vocal fold instability) |
| `shimmer_mean, std` | Amplitude cycle-to-cycle perturbation (%) | Elevated in AD/aging (motor control loss) |
| `hnr_mean, std, min, max` | Harmonics-to-noise ratio (dB) | Reduced in AD (noisier phonation) |

## Implementation Rationale

### Why These Features?

Pause timing was identified as the single most discriminative feature by Pérez-Toro et al. (2022) using ForestNet feature importance analysis on 229 Pitt Corpus recordings. This finding aligns with clinical understanding: patients with AD experience word-finding difficulty due to semantic memory decline, manifesting as longer and more frequent pauses. The feature generalizes across languages, reducing confounds from dialectal variation.

Prosodic features (F₀ range reduction, intensity flattening) map directly to documented voice changes in AD: reduced emotional expressiveness and monotonous speech quality. These reflect both cognitive (reduced linguistic planning) and motor deficits (reduced vocal control).

Voice quality features (jitter, shimmer, HNR) measure phonatory precision. While partly overlapping with normal aging effects, they capture additional variance in AD populations. HNR particularly reflects the vocal strain and breathiness reported in advanced AD speech samples.

The 36-feature set was chosen as a minimal sufficient set. Barragán Pulido et al. (2020) observed that feature sets >50 often show diminishing returns or worse performance due to overfitting; reducing to interpretable features below 40 improves both accuracy and clinical usability.

## Dataset: Pitt Corpus

The pipeline operates on the Pitt Corpus subset of DementiaBank (DOI: 10.21415/CQCW-1F92), which contains:

- **229 recordings** from the Cookie Theft picture description task (Boston Diagnostic Aphasia Examination)
- **139 AD patients**, **90 healthy controls**
- Average duration: 60–61 seconds per recording
- Demographic data: age, gender, MMSE scores, diagnostic labels

The Cookie Theft task stresses multiple cognitive domains (semantic memory, narrative planning, syntax), making speech impairments highly visible compared to less-demanding tasks.

## Acoustic Signal Processing Pipeline

### VAD Implementation

Speech/pause segmentation uses energy-based Voice Activity Detection (adapted from the cross-linguistic reference implementation, Pérez-Toro et al., 2025):

1. **Resampling**: Audio resampled to 16 kHz (standard for speech processing)
2. **Framing**: 25 ms windows with 10 ms hop
3. **Energy computation**: Short-time energy per frame
4. **Thresholding**: Frames classified as speech or pause based on energy threshold relative to noise floor

This VAD approach is robust to common speech audio conditions (background noise, variable recording quality) while remaining lightweight and deterministic—no learned models required.

### F₀ Extraction

Primary method: **pYIN** (Mauch & Dixon, 2014), a refined implementation of the YIN autocorrelation method. Extracts F₀ from voiced frames at 10 ms resolution. Falls back to parselmouth (Praat interface) if pYIN unavailable, ensuring robustness.

F₀ range: 50–500 Hz, covering typical male and female speakers. Unvoiced frames excluded from F₀ statistics.

### Voice Quality Computation

**Jitter/Shimmer**: Computed on F₀-extracted periods using cycle-to-cycle perturbation analysis. Parselmouth provides the underlying Praat algorithms.

**HNR**: Ratio of harmonic to noise energy, computed using spectral analysis. Higher values indicate more periodic (less noisy) vocalization.

## Performance Benchmarks

### Timing Features Alone
- **Pérez-Toro et al. (2022)**, Pitt Corpus (n=229): 69% UAR using ForestNet
- **Barragán Pulido et al. (2020)** meta-review: 75.2% accuracy (timing + energy features, SVM)
- **Cross-language (English→Spanish)**: AUC 0.75 on pause timing alone (Pérez-Toro et al., 2025)

### Combined Feature Set (Duration + Prosody + Voice Quality)
- **Pérez-Toro et al. (2022)**, Pitt Corpus: 79% UAR when all three families combined
- **Barragán Pulido reference systems**: 90.7% accuracy (timing + prosody + voice quality + SVM on AZTIAHO database)

These results establish that interpretable acoustic features, when properly extracted and combined, achieve clinically actionable performance without opaque embeddings.

## Installation and Setup

cd audio_feature_extraction

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt

### Dependencies

- librosa ≥ 0.10.0 — Audio I/O and pYIN F₀ extraction
- scipy ≥ 1.7.0 — Signal processing, statistics
- numpy ≥ 1.20.0 — Numerical operations
- parselmouth ≥ 0.4.0 — Praat interface (for HNR, jitter, shimmer fallback)
- soundfile ≥ 0.12.0 — WAV file handling
- pandas ≥ 1.3.0 — Data output and CSV writing
- pyyaml ≥ 5.4.0 — Configuration file parsing
- tqdm ≥ 4.60.0 — Progress bars

## Usage

### Single Recording

from audio_feature_extraction import extract_audio_features

features = extract_audio_features(
    audio_path='path/to/recording.wav',
    participant_id='001',
    group='AD',  # or 'HC'
    age=72,
    mmse=18
)

print(f"Pause ratio: {features['pause_ratio']:.3f}")
print(f"F0 range: {features['f0_range']:.1f} Hz")
print(f"HNR mean: {features['hnr_mean']:.2f} dB")

Returns dict with ~36 features.

### Batch Processing (Full Corpus)

from audio_feature_extraction import process_pitt_corpus

df = process_pitt_corpus(
    corpus_root='../Pitt Corpus/Pitt Corpus',
    output_csv='./output/pitt_audio_features.csv',
    task='cookie'
)

print(f"Processed {len(df)} recordings")
print(df.groupby('group')[['pause_ratio', 'f0_range', 'hnr_mean']].mean())

Outputs CSV with columns:
`participant_id, session, group, age, mmse, gender, duration_seconds, [36 features]`

### Command Line

python -m audio_feature_extraction.main_extractor \
    --corpus-root "../Pitt Corpus" \
    --output "./output/features.csv" \
    --task cookie

python -m audio_feature_extraction.main_extractor --list-features

## Directory Structure

audio_feature_extraction/
├── main_extractor.py          # Batch processing pipeline
├── vad_features.py            # VAD-based timing extraction
├── prosody_features.py        # F0 and intensity extraction
├── voice_quality_features.py  # Jitter, shimmer, HNR
├── utils/
│   ├── audio_loader.py        # Audio I/O and preprocessing
│   ├── pitt_metadata.py       # CHAT transcript parsing
│   └── validation.py          # Feature QC checks
├── config.yaml                # Processing parameters
├── tests/
│   └── test_features.py       # Unit tests
├── requirements.txt
└── README.md

## Configuration

Edit `config.yaml` to customize parameters:

audio:
  target_sample_rate: 16000    # Hz
  normalize: true              # RMS normalization

vad:
  window_size: 0.025           # 25 ms frames
  hop_size: 0.01               # 10 ms hop
  energy_threshold_db: 40      # dB below peak

prosody:
  f0_min: 50                   # Hz
  f0_max: 500                  # Hz
  f0_method: pyin              # pyin or parselmouth
  voicing_threshold: 0.1       # pYIN confidence

voice_quality:
  hnr_min_pitch: 75            # Hz (lower bound for HNR analysis)

## Expected Output Format

CSV file with columns:
participant_id,session,group,age,mmse,gender,duration_seconds,
pause_ratio,speech_ratio,pause_dur_mean,pause_dur_std,...
[36 additional features]

Example rows:
001,0,AD,57,18,M,54.76,0.32,0.68,1.15,...
002,0,HC,58,30,F,61.49,0.15,0.85,0.42,...

## Clinical Interpretation Guide

### Pause Metrics

Elevated pause ratio and duration reflect word-finding difficulty due to semantic memory decline—a hallmark of AD. The longest pause (pause_dur_max) is particularly discriminative, often exceeding 2–3 seconds in AD speakers compared to <1 second in controls.

### Prosodic Metrics

Reduced F₀ range (f0_range) maps to monotone, flattened speech—a consequence of reduced emotional expression and vocal motor control. F₀ itself is lower in elderly populations but AD patients show *further reduction in range*. Similarly, reduced intensity range reflects flat affect.

### Voice Quality Metrics

Elevated jitter and shimmer indicate pitch and amplitude instability, respectively. These reflect vocal fold stiffness or tremor, both common in AD. HNR reduction indicates breathy, noisy phonation. Note: jitter and shimmer also increase with normal aging, but combined with other features and reduced pitch range, they support AD classification.

## Processing Notes

- **Frame alignment**: VAD operates at 10 ms hops; F₀ and voice quality features aligned to the same timeline
- **Unvoiced frames**: Excluded from F₀ statistics; voice_rate metric captures voicing density
- **Extrema (min/max)**: Computed over entire recording; useful for detecting outlier events
- **Statistical functionals**: Mean, std, min, max, skewness, kurtosis computed per recording (not frame-level)

## Data Quality Checks

The pipeline performs QC during extraction:

1. Audio duration ≥30 seconds (insufficient speech otherwise)
2. F₀ frames ≥30% voiced (requires adequate phonation)
3. Pause ratio 0–1 (logical bounds)
4. HNR ≥0 dB (energy constraint)

Recordings failing any check are logged; features are set to NaN for manual review.

## References

[1] Barragán Pulido, M. L., et al. (2020). Alzheimer's disease and automatic speech analysis: A review. *Expert Systems with Applications*, 150, 113213.

[2] Pérez-Toro, P. A., et al. (2022). Interpreting acoustic features for the assessment of Alzheimer's disease using ForestNet. *Smart Health*, 26, 100347.

[3] Pérez-Toro, P. A., et al. (2025). Automated speech markers of Alzheimer dementia: A test of cross-linguistic generalizability. *Journal of Medical Internet Research*, 27, e74200.

[4] Mauch, M., & Dixon, S. (2014). pYIN: A fundamental frequency estimator using probabilistic threshold distributions. In *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

[5] DementiaBank Pitt Corpus. Retrieved from talkbank.org/dementia/access/English/Pitt.html. DOI: 10.21415/CQCW-1F92.