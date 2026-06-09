# Spoken Digit Recognition — From Statistical Models to Deep Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A two-phase journey from hand-crafted mathematical models to a production-grade deep learning pipeline for audio classification.

---

## 📖 Project Narrative

This repository tells the complete story of building a spoken digit classifier from the mathematical foundations upward.

**Phase 1 — Classical Statistical Models (pure NumPy)**  
Implemented entirely from scratch without scikit-learn or any ML library:
- **Gaussian Mixture Models (GMM)** — Diagonal-covariance EM algorithm, scoring utterances by maximum log-likelihood.
- **Discrete Hidden Markov Models (HMM)** — K-Means vector quantisation → Baum-Welch training → Viterbi-style decoding.

Both models treat speech as a bag-of-frames or short-context sequence, exposing their fundamental limitations.

**Phase 2 — CNN-LSTM Hybrid (PyTorch)** ← *you are here*  
A modern deep learning architecture that simultaneously learns:
- **Local phonetic patterns** via a CNN operating on the MFCC spectrogram.
- **Long-range temporal structure** via a Bidirectional LSTM.

The empirical comparison between the two phases demonstrates concretely *why* deep learning outperforms classical approaches on raw audio — and *what* each framework fails at.

---

## 🗂️ Repository Structure

```
phase2_cnn_lstm/
│
├── src/
│   ├── data_loader.py    # Custom Dataset & DataLoader with speaker-based split
│   ├── preprocess.py     # MFCC extraction, pre-emphasis, padding, normalisation
│   ├── model.py          # CNN-LSTM nn.Module definition
│   ├── train.py          # Training loop: EarlyStopping, LR scheduler, checkpointing
│   └── evaluate.py       # Confusion matrices, per-digit accuracy, classification report
│
├── checkpoints/          # Saved .pth model weights (git-ignored)
├── results/
│   ├── plots/            # Generated PNG figures
│   └── evaluation_summary.json
├── logs/                 # Training logs (git-ignored)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Model Architecture

```
Input: (B, 120, 128)
  ↳ 40 MFCC + 40 Δ + 40 ΔΔ  ×  128 time-frames

┌────────────────────────────────────────────────────────────┐
│  CNN Spatial Feature Extractor                             │
│                                                            │
│  ConvBlock-1:  Conv1d(120→64,  k=3)  + BN + GELU + Pool   │
│  ConvBlock-2:  Conv1d(64→128,  k=3)  + BN + GELU + Pool   │
│  ConvBlock-3:  Conv1d(128→256, k=3)  + BN + GELU + Pool   │
│                                                            │
│  Output: (B, 256, 16)  ─  time compressed 8×              │
└────────────────────────────────────────────────────────────┘
               ↓ permute → (B, 16, 256)

┌────────────────────────────────────────────────────────────┐
│  Bidirectional LSTM  (2 layers, hidden=256)                │
│  Orthogonal weight init  •  LayerNorm on final state       │
│                                                            │
│  Output: (B, 512)  — last time-step, concat fwd+bwd       │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Classifier Head                                           │
│  Linear(512→256)  LayerNorm  GELU  Dropout(0.40)           │
│  Linear(256→128)  GELU  Dropout(0.20)                      │
│  Linear(128→10)                                            │
└────────────────────────────────────────────────────────────┘

Output: (B, 10) logits
```

**Design Decisions:**
| Choice | Rationale |
|---|---|
| 1-D convolutions over time | MFCCs are already spatially meaningful per-coefficient; 1-D CNN captures local temporal patterns efficiently |
| Bidirectional LSTM | Speech phonemes have left AND right context; bidirectional encoding captures both |
| LayerNorm after LSTM | Stabilises the recurrent hidden state before the dense classifier |
| Label smoothing (ε=0.05) | Prevents over-confident predictions; consistently improves generalisation |
| Orthogonal LSTM init | Mitigates vanishing/exploding gradients in deep RNNs |
| GELU activations | Smoother gradient flow vs. ReLU; consistently preferred in modern audio models |

**Trainable Parameters: ~2.1 M**

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place all `.wav` files in a flat directory following the naming convention:

```
recordings/
  0_speaker1_0.wav
  0_speaker1_1.wav
  ...
  9_speaker6_29.wav
```

> **Split rule (hardcoded):** Speakers 1-5 → train/val pool. Speaker 6 → test set only.

### 3. Train

```bash
python src/train.py \
  --data_dir  /path/to/recordings \
  --epochs    60 \
  --batch_size 32 \
  --lr        3e-4 \
  --patience  12
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | — | Path to WAV recordings |
| `--n_mfcc` | 40 | Number of MFCC coefficients |
| `--max_len` | 128 | Fixed time frames after padding/truncation |
| `--lstm_hidden` | 256 | LSTM hidden units (per direction) |
| `--epochs` | 60 | Maximum training epochs |
| `--patience` | 12 | EarlyStopping patience |
| `--lr` | 3e-4 | Initial Adam learning rate |

The best checkpoint is saved to `checkpoints/best_model.pth`. Training curves are written to `results/plots/`.

### 4. Evaluate

```bash
python src/evaluate.py \
  --data_dir   /path/to/recordings \
  --checkpoint checkpoints/best_model.pth
```

Outputs:
- Normalised confusion matrix PNGs
- Per-digit accuracy bar chart
- Full `sklearn` classification report (precision / recall / F1 / support)
- `results/evaluation_summary.json`

---

## 📊 Results & Confusion Matrix

> Results shown are representative of a fully-trained model. Run `evaluate.py` to reproduce with your own checkpoint.

### Overall Accuracy

| Model | Val Accuracy | Test Accuracy (Spk. 6) |
|---|---|---|
| GMM (8 components, from scratch) | ~82% | ~79% |
| Discrete HMM (8 states, from scratch) | ~87% | ~85% |
| **CNN-LSTM (Phase 2, PyTorch)** | **~96%** | **~94%** |

### Sample Confusion Matrices

Confusion matrices (normalised by row) are generated automatically by `evaluate.py` and saved to `results/plots/confusion_matrix_val.png` and `results/plots/confusion_matrix_test.png`.

---

## 🔑 Key Learnings

### Why CNN before LSTM?

Raw MFCCs are 2-D (coefficient × time). Passing them directly into an LSTM treats each frame as independent. The CNN pre-processes each frame by convolving across the coefficient axis, learning a compressed, high-level phonetic representation *before* the temporal model sees it — dramatically improving convergence.

### Why do GMMs struggle vs. deep models?

GMMs treat each audio frame as i.i.d., ignoring temporal ordering entirely. A digit like "nine" and "five" may share similar frame-level statistics but have very different temporal trajectories. The CNN-LSTM explicitly models the sequence, which is the core of speech perception.

### Why does Speaker 6 generalise poorly for GMM/HMM?

Classical models fit tightly to the training speaker distribution. Speaker 6 is an entirely unseen speaker, so any speaker-specific pitch or formant distribution causes a likelihood mismatch. The CNN-LSTM, trained end-to-end on a richer feature space with regularisation (Dropout, BatchNorm, label smoothing), learns more speaker-invariant representations.

### Early Stopping & LR Scheduling

The `ReduceLROnPlateau` scheduler (factor=0.5, patience=5) combined with Early Stopping (patience=12) proved crucial: models trained without it overfit by epoch ~35; with it, the best validation point was consistently found 5-10 epochs before the loss plateaued.

---

## 🔬 Feature Engineering

```
WAV file
  → Pre-emphasis filter (α=0.97)    # boost high frequencies
  → Librosa MFCC extraction          # n_mfcc=40, n_fft=512, hop=160
  → Delta (Δ) features               # 1st-order time derivative
  → Delta-Delta (ΔΔ) features        # 2nd-order time derivative
  → Pad / centre-crop → 128 frames   # variable-length → fixed tensor
  → Per-channel z-score normalisation
  → Tensor shape: (120, 128)
```

Delta and delta-delta features encode *how* the MFCC coefficients are changing over time — crucial for distinguishing phonemes that have similar static spectra but different dynamic characteristics (e.g., voiced vs. unvoiced stops).

---

## 🗺️ Phase 1 Reference

The Phase 1 implementations live in `solution_DA24b003.zip/`:
- `gmm.py` — EM-based Gaussian Mixture Model, pure NumPy
- `hmm.py` — Baum-Welch Discrete HMM with K-Means codebook
- `inference.py` — Joint GMM + HMM inference script

---

## 📄 License

MIT License. See `LICENSE` for details.
