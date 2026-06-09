<div align="center">

# 🎙️ Classical vs. Deep Speech Recognition

**A Comparative Implementation of Statistical Learning (GMM/HMM) and Deep Learning (CNN-LSTM) for Spoken Digit Classification.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)](https://scipy.org/)


</div>

## 📌 Project Overview

This repository features a comprehensive end-to-end framework for **isolated spoken digit recognition (0-9)**. It benchmarks traditional statistical machine learning techniques against modern deep neural architectures. The project rigorously compares **Gaussian Mixture Models (GMM)** and **Hidden Markov Models (HMM)** with a hybrid **CNN-LSTM Neural Network** built in PyTorch. 

By enforcing a strict zero-leakage training protocol (training on Speakers 1-5, testing on Speaker 6), this project demonstrates robust cross-speaker generalization, advanced feature extraction, and production-ready PyTorch development practices.

---

## 🚀 Key Features & Methodologies

### 1. Feature Engineering
* **MFCC Extraction:** Transforming raw audio waveforms into Mel-Frequency Cepstral Coefficients (MFCCs) to capture the human auditory system's response.
* **Temporal Dynamics:** Computing delta and delta-delta features to capture speech transitions over time.

### 2. Phase 1: Statistical Machine Learning (Classical ML)
* **Gaussian Mixture Models (GMM):** Implemented to model the acoustic feature distributions probabilistically.
* **Hidden Markov Models (HMM):** Designed to capture the temporal sequential nature of speech, mapping state transitions to phonemic utterances.
* **Inference Pipeline:** Custom algorithmic inference scripts for probabilistic sequence evaluation.

### 3. Phase 2: Deep Learning (CNN-LSTM Architecture)
* **Hybrid Neural Architecture:** 
  * **CNN Backbone:** Extracts localized spatial acoustic features directly from the 2D MFCC spectrograms.
  * **LSTM Sequence Modeling:** Models the long-term temporal dependencies of the extracted feature maps.
* **Strict Speaker-Split DataLoader:** Custom PyTorch `Dataset` enforcing zero data leakage by isolating Speaker 6 entirely for evaluation.
* **Robust Training Pipeline:** Implements dynamic learning rate scheduling, dropout regularization, and early stopping.

---

## 📊 Dataset & Evaluation Setup

* **Domain:** Isolated spoken digits (0 through 9).
* **Data Split Strategy:**
  * **Training Set:** Utterances from Speakers 1 through 5.
  * **Test Set:** Utterances from Speaker 6.
  * *Rationale:* This rigorous split ensures the models learn generalized phonetic representations rather than memorizing speaker-specific voice timbres.

---

## 🛠️ Repository Structure

```text
├── phase1_classical_ml/    # GMM and HMM implementations and notebooks
├── phase2_deep_learning/   # PyTorch CNN-LSTM implementation
└── docs/                   # Project specifications
