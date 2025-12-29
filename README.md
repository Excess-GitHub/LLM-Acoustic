# LLM-Acoustic: AD Detection with Fine-Tuned LLMs and Interpretable Audio Features

## Project Overview

This repository addresses early detection of Alzheimer's disease (AD) and mild cognitive impairment (MCI) from spontaneous speech using a hybrid approach that combines fine-tuned large language models (LLMs) with interpretable acoustic features. The core motivation is to evaluate whether clinically interpretable audio features can achieve competitive performance compared to opaque embedding-based approaches while remaining explainable to healthcare practitioners.

The project builds on two key findings:

1. **Casu et al. (2024)** demonstrated that integrating LLM-based linguistic features with VGGish-derived acoustic embeddings achieved ~93% accuracy on the ADReSSo benchmark (Mistral-7B + VGGish + SVC).

2. **Cross-linguistic timing studies** (Pérez-Toro et al., 2025) showed that pause-based and prosodic features generalize well across English and Spanish with direct clinical interpretability: pause duration maps to word-finding difficulty, reduced F0 range to monotonous speech, etc.

## Research Questions

**Primary Question:** How much do acoustic features improve AD detection when added to a strong LLM-only model on the Pitt Corpus?

**Secondary Question:** Can we restrict ourselves to interpretable audio features (pause timing, prosody, voice quality) without sacrificing performance compared to opaque embedding-based features like VGGish?

The answer has important implications for clinical adoption: if interpretable features achieve 90% of the performance at 10× the explainability, that trade-off is clinically valuable.

## Datasets: Pitt Corpus and the Cookie Theft Task

The project uses the **Pitt Corpus** from DementiaBank, which contains English-speaking participants describing the "Cookie Theft" picture from the Boston Diagnostic Aphasia Examination. The corpus includes:

- **AD/Dementia patients** and **healthy controls (HC)**
- Spontaneous speech recordings (WAV/MP3 format)
- CHAT-format transcripts with speaker metadata (age, gender, MMSE scores)
- **Sample sizes:** 166 training participants (87 AD, 79 HC); 71 test participants (35 AD, 36 HC)

### Why the Cookie Theft Task?

The picture description task is valuable for AD screening because it stresses several cognitive functions that decline early in dementia:

- **Semantic memory** — retrieving words for objects in the scene
- **Narrative organization** — structuring a coherent description
- **Discourse-level planning** — knowing what to say and in what order

These deficits manifest as measurable speech changes:
- Longer pauses (word retrieval difficulty)
- Reduced lexical diversity
- Simpler syntax
- Flattened prosody

### Data Layout

```
Pitt Corpus/
├── Media/
│   ├── Dementia/
│   │   └── Cookie/
│   │       └── WAV/
│   │           ├── 001-0.wav
│   │           └── ...
│   └── Control/
│       └── Cookie/
│           └── WAV/
└── Transcripts/
    └── Pitt/
        ├── Dementia/
        │   └── cookie/
        │       ├── 001-0.cha
        │       └── ...
        └── Control/
            └── cookie/
```

## Repository Structure

```
LLM-Acoustic/
├── README.md                        # This file
├── Timing_VAD.py                    # Reference: cross-linguistic VAD implementation
│
├── audio_feature_extraction/        # Interpretable feature extraction pipeline
│   ├── vad_features.py              # Pause/speech timing extraction
│   ├── prosody_features.py          # F0 and intensity statistics
│   ├── voice_quality_features.py    # Jitter, shimmer, HNR
│   ├── main_extractor.py            # Batch extraction script
│   ├── config.yaml                  # Extraction parameters
│   ├── output/
│   │   └── pitt_audio_features.csv  # ~36 interpretable features per recording
│   └── README.md                    # Feature documentation
│
├── ad_detection/                    # LLM + acoustic feature experiments
│   ├── train.py                     # Main training script
│   ├── compare_features.py          # Compare LLM-only vs VGGish vs interpretable
│   ├── config.yaml                  # Model and training parameters
│   ├── src/
│   │   ├── llm_features.py          # LLM fine-tuning and embedding extraction
│   │   ├── vggish_features.py       # VGGish feature extractor
│   │   ├── gru_autoencoder.py       # GRU-AE for VGGish compression
│   │   ├── feature_fusion.py        # Multimodal fusion + feature selection
│   │   ├── classifier.py            # SVC, RF, XGB classifiers
│   │   └── utils.py
│   ├── output/
│   │   └── results.json             # Metrics by fold and configuration
│   └── README.md                    # Experimental design
│
└── Pitt Corpus/                     # (Not in repo) Local copy of DementiaBank data
```

## Interpretability vs. Performance: The Trade-Off

A central theme of this project is understanding the trade-off between **feature interpretability** and **classification performance**. We are not trying to match state-of-the-art; rather, we are quantifying the cost of interpretability.

### Two Feature Families

**Embedding-based / Black-box Features:**
- VGGish, Wav2Vec, WavLM, OpenSmile/GeMAPS-derived representations
- High-dimensional, learned representations
- Best raw performance but opaque to clinicians

**Hand-crafted, Interpretable Features:**
- Pause timing: pause ratio, duration mean/std, speech segment statistics
- Prosody: F0 mean/range, intensity variation, voice rate
- Voice quality: jitter, shimmer, harmonics-to-noise ratio
- Directly map to clinical constructs

### Benchmark Performance

| Approach | Dataset | Performance |
|----------|---------|-------------|
| ADScreen (acoustic only) | Pitt Corpus | 78.87% accuracy |
| ADScreen (acoustic + linguistic) | Pitt Corpus | 83.09% accuracy |
| ADScreen (all components) | Pitt Corpus | 90.14% accuracy, AUC 0.939 |
| Timing features (within-language) | English | AUC 0.79 |
| Timing + lexico-semantic (within-language) | English | AUC 0.88 |
| Timing features (cross-language) | English→Spanish | AUC 0.75 |
| VGGish embeddings | Various | 64–73% accuracy |
| LLM-only baseline (expected) | Pitt Corpus | ~88–90% |

### What We're Measuring

1. Extract ~36 interpretable features (timing, prosody, voice quality)
2. Compare performance of:
   - **LLM-only** (Mistral-7B fine-tuned on transcripts)
   - **LLM + VGGish** (following Casu et al.'s approach)
   - **LLM + Interpretable features** (from our pipeline)
3. Document whether the performance cost is acceptable for clinical use

Expected results:
- **LLM-only:** ~88–90% accuracy
- **LLM + VGGish:** ~91–93% (2–3% improvement)
- **LLM + Interpretable:** Somewhere in between, with question being how close to VGGish while remaining fully explainable

## Getting Started

### 1. Add Dataset
- Download the Pitt Corpus dataset from Data/DementiaBank in UIC Box
- Place the downloaded Pitt Corpus dataset in the Pitt Corpus folder
- The final structure should be
```
Pitt Corpus/Pitt Corpus/
├── Media                      
├── Transcripts
```

### 2. Extract Interpretable Audio Features

```bash
cd audio_feature_extraction
pip install -r requirements.txt

python main_extractor.py \
    --corpus-root "../Pitt Corpus/Pitt Corpus" \
    --output "./output/pitt_audio_features.csv" \
    --task cookie
```

This generates a CSV with ~36 features per recording.

### 3. Run AD Detection Experiments

```bash
cd ad_detection
pip install -r requirements.txt

# Compare all feature types
python compare_features.py --config config.yaml --n-folds 5

# Or train specific configuration
python train.py --config config.yaml --use-custom-acoustic
```

### 4. View Results

Results are saved to `ad_detection/output/` as JSON files with per-fold metrics.

## Acoustic Features (~36 Total)

Extracted using OpenSMILE and PRAAT, organized into clinical domains:

### Pause/Speech Timing (Word-Finding Difficulty)
- Pause ratio, pause duration statistics
- Speech segment metrics
- Voice rate
- Silence indicators

### Prosody (Monotone Speech)
- Fundamental frequency (F0) statistics
- F0 variation metrics
- Jitter (pitch perturbation)

### Voice Quality (Vocal Control)
- Shimmer (amplitude perturbation)
- Harmonic-to-noise ratio
- Voice-breaking indicators
- Intensity statistics

### Rhythmic Structure
- Pairwise Variability Index
- Syllabic duration variation
- Sentence rhythm

Each feature maps to documented AD symptoms:
- **Pause duration** ↔ semantic memory impairment
- **F0 range reduction** ↔ monotone speech
- **Shimmer increase** ↔ vocal instability
- **Speech rate decrease** ↔ cognitive processing delay

## Related Work

### ADScreen (Zolnoori et al., 2023)

The most comprehensive interpretable AD detection system:

- **Acoustic component:** 78.87% accuracy (5 domains)
- **Linguistic component:** 83.09% accuracy (pausing, lexical richness, syntax, embeddings)
- **Psycholinguistic component:** GeMAPS + LIWC features
- **Combined:** 90.14% accuracy, AUC 0.939

Key insight: High performance is achievable with interpretable features when acoustic, linguistic, and psycholinguistic modalities are combined.

### Cross-Linguistic Timing Study (Pérez-Toro et al., 2025)

Tested pause-based timing features across English and Spanish:

- **Within-language:** AUC 0.88 with combined timing + lexico-semantic
- **Cross-language:** AUC 0.75 with timing alone
- **Key finding:** Timing features generalize cross-linguistically

This supports our focus on timing features as language-agnostic and interpretable.

## Why Interpretability Matters

Clinical adoption requires:

1. **Explainability:** Clinicians must understand why a system suggests AD
2. **Debuggability:** When performance differs, we need to understand why
3. **Regulatory compliance:** FDA increasingly requires interpretable models
4. **Trust:** Clinicians act on recommendations they understand

Black-box models fail these requirements. Interpretable features succeed because they are grounded in AD pathophysiology.

## Notes

Key implementation details are documented in respective README files:
- `audio_feature_extraction/README.md` — Feature definitions and clinical rationale
- `ad_detection/README.md` — Experimental design and results reporting

## References

[1] Zolnoori, M., Zolnour, A., & Topaz, M. (2023). ADscreen: A speech processing-based screening system for automatic identification of patients with Alzheimer's disease and related dementia. *Artificial Intelligence in Medicine*, 143, 102624.

[2] Pérez-Toro, P. A., et al. (2025). Automated speech markers of Alzheimer dementia: A test of cross-linguistic generalizability. *Journal of Medical Internet Research*, 27, e74200.

[3] Casu, F., et al. (2024). Integrating fine-tuned LLM with acoustic features for enhanced detection of Alzheimer's disease. *IEEE Access*.

[4] Barragán Pulido, M. L., et al. (2020). Alzheimer's disease and automatic speech analysis: A review. *Expert Systems with Applications*, 150, 113213.

[5] DementiaBank Pitt Corpus. DOI 10.21415/CQCW-1F92