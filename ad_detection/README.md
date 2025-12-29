# AD Detection with Fine-Tuned LLM and Acoustic Features

Adapts the Casu et al. (2024) multimodal framework to the Pitt Corpus for comparing interpretable acoustic features against standard embedding-based approaches.

## Scientific Motivation

Two research questions drive this implementation:

1. **Feature modality comparison**: How do interpretable acoustic features (pause timing, prosody, voice quality) compare to opaque embedding-based features (VGGish) when combined with LLM-derived linguistic representations?

2. **Performance-interpretability trade-off**: What accuracy cost results from using clinically interpretable features instead of high-dimensional embeddings?

Casu et al. (2024) demonstrated on ADReSSo 2021 that fine-tuned Mistral-7B combined with VGGish acoustic embeddings achieves 93.0% accuracy, a +1.5% improvement over linguistic features alone (91.5%). However, the relative contribution of interpretable versus opaque acoustic features remains unclear on other datasets. This implementation addresses that gap using the Pitt Corpus.

## Experimental Design

### Data

- **Dataset**: Pitt Corpus Cookie Theft recordings (DementiaBank, DOI: 10.21415/CQCW-1F92)
- **Sample size**: 229 recordings (139 AD, 90 healthy controls)
- **Task**: Cookie Theft picture description (Boston Diagnostic Aphasia Examination)
- **Feature configuration**:
  - **LLM features**: 4096-dimensional embedding from fine-tuned Mistral-7B
  - **Acoustic features** (three conditions):
    - None (LLM-only baseline)
    - VGGish embeddings (128-d per 0.96s window) → 64-d via GRU autoencoder
    - Interpretable features from `../audio_feature_extraction/` (~36 features)

### Models

Following Casu et al.'s methodology:

#### LLM-X: Linguistic-Only Baseline

- Fine-tune Mistral-7B on Cookie Theft transcripts using LoRA (r=20, alpha=40, dropout=0.01)
- Extract final hidden layer → mean-pool → 4096-d feature vector
- Train Support Vector Classifier (SVC with RBF kernel, C=1.0)

#### LLM-A-X (VGGish): Standard Embedding Multimodal

- Same Mistral-7B features
- **Acoustic**: VGGish embeddings → 64-d via GRU autoencoder (trained jointly with classification)
- **Fusion**: Concatenate LLM (4096-d) + acoustic (64-d), apply LinearSVC with L1 penalty for feature selection (95% cumulative importance)
- Replicates Casu et al.'s exact approach

#### LLM-A-X (Interpretable): Clinically-Motivated Features

- Same Mistral-7B LLM features
- **Acoustic**: Interpretable features from `../audio_feature_extraction/`:
  - Pause timing: pause_ratio, pause_dur_mean/std/min/max/skew/kurt, speech segments (17 features)
  - Prosody: F₀ mean/std/range, intensity statistics, voice_rate (11 features)
  - Voice quality: jitter, shimmer, HNR (8 features)
  - Total: ~36 features
- **Same fusion pipeline** as VGGish for fair comparison

### Training Protocol

- **Cross-validation**: 5-fold stratified
- **Per-fold**: (1) Split data, (2) Fine-tune LLM on training fold, (3) Extract features, (4) Train classifier
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Robustness**: Each configuration repeated 25× with different random seeds (Casu et al. methodology)
- **Hyperparameters**: Use Casu et al.'s published values by default (Table III); Optuna available for tuning

### Expected Performance

**ADReSSo 2021 (Casu et al., 2024):**
- LLM-only: 91.5% accuracy
- LLM + VGGish: 93.0% accuracy (+1.5%)

**Pitt Corpus expectations:**
- Pérez-Toro et al. (2022): Timing alone 69% UAR, combined features 79% UAR (n=229, same corpus)
- Pérez-Toro et al. (2025): Cross-linguistic timing AUC 0.75, within-language combined AUC 0.88
- Hypothesis:
  - LLM-only: ~88-92% (strong baseline)
  - LLM + Interpretable: ~88-90%
  - LLM + VGGish: ~90-93% (expected +2-3% over interpretable)

## Code Organization

```
ad_detection/
├── config.yaml
├── train.py
├── compare_features.py
├── src/
│   ├── data_loader.py
│   ├── llm_features.py
│   ├── vggish_features.py
│   ├── gru_autoencoder.py
│   ├── feature_fusion.py
│   ├── classifier.py
│   └── trainer.py
├── output/
│   └── results.json
├── checkpoints/
└── cache/
```

## Quick Start

### Prerequisites

```bash
cd ad_detection
pip install -r requirements.txt
huggingface-cli login
```

### Run Feature Comparison

```bash
python compare_features.py --config config.yaml --n-folds 5
```

For GPU with <8GB VRAM:

```bash
python compare_features.py --config config.yaml --n-folds 5 --small-gpu
```

### Train Individual Models

```bash
# LLM-only
python train.py --config config.yaml

# LLM + interpretable acoustic
python train.py --config config.yaml --use-custom-acoustic

# Small GPU (uses TinyLlama)
python train.py --config config.yaml --small-gpu --use-custom-acoustic
```

## Configuration

Key settings in `config.yaml`:

```yaml
# LLM (from Casu et al. Table III)
llm:
  model_name: "mistralai/Mistral-7B-v0.1"
  lora:
    r: 20
    alpha: 40
    dropout: 0.01

# Training
training:
  epochs: 5
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5

# Classifier
classifier:
  type: "svc"
  svc:
    kernel: "rbf"
    C: 1.0

# Feature fusion
feature_selection:
  method: "linear_svc_l1"
  importance_threshold: 0.95
```

## Results

### Preliminary Results (Single Fold, TinyLlama-1.1B)

**Important Limitations:** These results are from a preliminary single-fold test using TinyLlama-1.1B (not Mistral-7B) due to GPU constraints. Robust evaluation requires full 5-fold cross-validation with Mistral-7B.

**Configuration:**
- Model: TinyLlama-1.1B-Chat-v1.0 (4-bit quantization)
- Split: 445 train / 103 test (from 548 total: 308 AD, 240 HC)
- Classifier: SVC with RBF kernel

| Approach | Accuracy | Precision | Recall | F1 | ROC-AUC |
|----------|----------|-----------|--------|-----|---------|
| LLM-only | 74.56% | 71.79% | 88.89% | 79.43% | 81.33% |
| LLM + VGGish | 74.56% | 71.79% | 88.89% | 79.43% | 81.85% |
| LLM + Interpretable | **77.19%** | **75.34%** | **87.30%** | **80.88%** | **82.48%** |

### Observations

1. **Interpretable features outperformed both baselines** in this single-fold test (+2.63% accuracy)
2. **LLM+VGGish showed minimal improvement** over LLM-only (same accuracy, +0.52% ROC-AUC)
3. **Single-fold variance**: Test set (103 samples, ~19% of data) provides unstable estimates

### Comparison with Casu et al. (2024)

Casu et al. on ADReSSo 2021 using Mistral-7B (5-fold CV):

| Approach | Casu et al. (Mistral-7B) | Our Preliminary (TinyLlama) |
|----------|--------------------------|------------------------------|
| LLM-only | 91.5% ± 4.3% | 74.56% |
| LLM + VGGish | **93.0% ± 4.1%** | 74.56% |

**Key differences:**
- Model size: Mistral-7B (7B parameters) vs TinyLlama (1.1B) → smaller model underperforms on linguistic understanding
- Dataset: ADReSSo (166 train, 71 test) vs Pitt (445 train, 103 test)
- Evaluation: 5-fold CV with 25 repetitions vs single fold

Casu et al. found VGGish provides +1.5% boost with Mistral-7B. Our preliminary test with TinyLlama showed no improvement, likely reflecting the smaller model's limited capacity.

### Requirements for Robust Results

To obtain reliable estimates comparable to Casu et al.:

1. **Use Mistral-7B** (requires GPU with ≥16GB VRAM)
2. **5-fold cross-validation** (current results from single fold)
3. **Multiple random seed repetitions** (Casu et al. used 25×)
4. **Full dataset** (ensure all 229 Pitt recordings included)

**Recommended:**
```bash
python compare_features.py --config config.yaml --n-folds 5
# (Without --small-gpu to use Mistral-7B)
```

### Interpretability Trade-Off (Preliminary)

In this single-fold test, interpretable features outperformed VGGish. While encouraging, this should be interpreted cautiously given the TinyLlama limitation and single-fold variance.

If this pattern holds with Mistral-7B and 5-fold CV:
- No performance cost for interpretability
- Strong justification for clinical deployment

More likely scenario (based on Casu et al. and literature):
- LLM+Interpretable: ~88-90%
- LLM+VGGish: ~90-93%
- A 2-3% gap potentially acceptable given interpretability benefits

## Key Insights from Literature

### Casu et al. (2024)

- Fine-tuned Mistral-7B significantly outperforms zero-shot LLM baselines
- VGGish acoustic embeddings provide +1.5% gain on ADReSSo
- SVC outperformed RF, XGB, ANN classifiers
- LoRA (r=20, alpha=40) achieves parameter efficiency and performance balance

### Pérez-Toro et al. (2022) on Pitt Corpus

- Pause timing features alone: 69% UAR (n=229)
- Combined timing + prosody + voice quality: 79% UAR
- Duration-based features identified as most discriminative

### Pérez-Toro et al. (2025) Cross-Linguistic

- Within-language (English): AUC 0.88 (timing + lexico-semantic)
- Cross-language: AUC 0.75 (timing alone, English→Spanish)
- Timing features generalize across languages; lexico-semantic features do not

### Barragán Pulido et al. (2020) Review

- Conventional acoustic features (pause, F₀, jitter, shimmer, HNR) achieve 70-90% across 90+ studies
- Feature sets >50 often show overfitting; <40 interpretable features preferable
- Combining complementary feature families improves performance while maintaining interpretability

## References

[1] Casu, F., et al. (2024). Integrating fine-tuned LLM with acoustic features for enhanced detection of Alzheimer's disease. *IEEE Journal of Biomedical and Health Informatics*, DOI: 10.1109/JBHI.2025.3566615.

[2] Pérez-Toro, P. A., et al. (2022). Interpreting acoustic features for the assessment of Alzheimer's disease using ForestNet. *Smart Health*, 26, 100347.

[3] Pérez-Toro, P. A., et al. (2025). Automated speech markers of Alzheimer dementia: A test of cross-linguistic generalizability. *Journal of Medical Internet Research*, 27, e74200.

[4] Barragán Pulido, M. L., et al. (2020). Alzheimer's disease and automatic speech analysis: A review. *Expert Systems with Applications*, 150, 113213.

[5] DementiaBank Pitt Corpus. DOI: 10.21415/CQCW-1F92.