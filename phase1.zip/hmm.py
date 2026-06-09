"""
hmm.py
------
Discrete Hidden Markov Model (HMM) for spoken-digit recognition.

The approach:
  1. Train a K-Means codebook on all training MFCC frames.
  2. Quantise every utterance to a symbol sequence.
  3. Train one DiscreteHMM per digit (0-9) with Baum-Welch.

Usage (training / evaluation):
    python hmm.py

The script expects the recordings folder at the path set in
RECORDINGS_FOLDER.  It saves the trained bundle (codebook + HMMs + mean/std)
to `trained_hmms.pkl` in the same directory.

Exported symbols used by inference.py:
    KMeans               – simple K-Means
    DiscreteHMM          – the HMM class
    predict_digit_dhmm   – classification helper
    load_hmms            – loads a previously saved bundle
"""

import os
import pickle
import random

import numpy as np

# Reuse feature / dataset helpers from gmm.py
from gmm import (
    prepare_dataset,
    extract_features,
    extract_for_inference,
    RECORDINGS_FOLDER,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_hmms.pkl")

N_CODEWORDS = 64

# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

class KMeans:
    def __init__(self, n_clusters=64, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state
        self.centroids    = None

    def fit(self, X):
        rng     = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(self.max_iter):
            labels = self._assign(X)
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0
                else self.centroids[k]
                for k in range(self.n_clusters)
            ])
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
        return self

    def _assign(self, X):
        diffs = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        return np.argmin(dists, axis=1)

    def predict(self, X):
        return self._assign(X)

# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def quantize_sequence(feature_matrix, kmeans):
    return kmeans.predict(feature_matrix)


def quantize_dataset(data, kmeans):
    quantized = []
    for item in data:
        quantized.append({
            "digit":    item["digit"],
            "speaker":  item["speaker"],
            "sequence": quantize_sequence(item["features"], kmeans),
        })
    return quantized

# ---------------------------------------------------------------------------
# Discrete HMM (Baum-Welch)
# ---------------------------------------------------------------------------

class DiscreteHMM:
    """Left-to-right discrete HMM trained with Baum-Welch."""

    def __init__(self, n_states, n_codewords, max_iter=100, tol=1e-4, random_state=42):
        self.n_states    = n_states
        self.n_codewords = n_codewords
        self.max_iter    = max_iter
        self.tol         = tol
        self.random_state = random_state

        self.pi = None  # initial state distribution
        self.A  = None  # transition matrix  (n_states × n_states)
        self.B  = None  # emission matrix    (n_states × n_codewords)

    # ------------------------------------------------------------------
    def _initialize(self):
        rng = np.random.default_rng(self.random_state)

        self.pi    = np.zeros(self.n_states)
        self.pi[0] = 1.0

        # Left-to-right topology
        self.A = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            if i == self.n_states - 1:
                self.A[i, i] = 1.0
            else:
                self.A[i, i]     = 0.5
                self.A[i, i + 1] = 0.5

        self.B  = rng.random((self.n_states, self.n_codewords)) + 1e-6
        self.B /= self.B.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    def _forward(self, obs):
        T     = len(obs)
        alpha = np.zeros((T, self.n_states))
        scale = np.zeros(T)

        alpha[0] = self.pi * self.B[:, obs[0]]
        scale[0] = alpha[0].sum()
        if scale[0] == 0:
            scale[0] = 1e-12
        alpha[0] /= scale[0]

        for t in range(1, T):
            alpha[t] = alpha[t - 1] @ self.A * self.B[:, obs[t]]
            scale[t] = alpha[t].sum()
            if scale[t] == 0:
                scale[t] = 1e-12
            alpha[t] /= scale[t]

        log_likelihood = np.sum(np.log(scale + 1e-12))
        return alpha, scale, log_likelihood

    def _backward(self, obs, scale):
        T    = len(obs)
        beta = np.zeros((T, self.n_states))
        beta[T - 1] = 1.0 / (scale[T - 1] + 1e-12)

        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])
            beta[t] /= (scale[t] + 1e-12)
        return beta

    # ------------------------------------------------------------------
    def fit(self, sequences):
        self._initialize()
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            total_ll = 0.0

            acc_A              = np.zeros((self.n_states, self.n_states))
            acc_B              = np.zeros((self.n_states, self.n_codewords))
            acc_gamma_sum      = np.zeros(self.n_states)
            acc_gamma_sum_excl = np.zeros(self.n_states)

            for obs in sequences:
                T             = len(obs)
                alpha, scale, ll = self._forward(obs)
                total_ll     += ll
                beta          = self._backward(obs, scale)

                gamma         = alpha * beta * scale.reshape(-1, 1)
                gamma_sum     = gamma.sum(axis=0)

                for t in range(T - 1):
                    numer   = (alpha[t].reshape(-1, 1)
                               * self.A
                               * self.B[:, obs[t + 1]]
                               * beta[t + 1])
                    denom   = numer.sum() + 1e-12
                    acc_A  += numer / denom

                acc_gamma_sum      += gamma_sum
                acc_gamma_sum_excl += gamma[:-1].sum(axis=0)

                for t in range(T):
                    acc_B[:, obs[t]] += gamma[t]

            if iteration > 0 and (total_ll - prev_ll) < self.tol:
                break
            prev_ll = total_ll

            row_sums = acc_gamma_sum_excl.reshape(-1, 1) + 1e-12
            self.A   = acc_A / row_sums
            self.A  /= (self.A.sum(axis=1, keepdims=True) + 1e-12)

            self.B  = acc_B / (acc_gamma_sum.reshape(-1, 1) + 1e-12)
            self.B /= (self.B.sum(axis=1, keepdims=True) + 1e-12)

        return self

    def score(self, obs):
        """Return normalised log-likelihood of an observation sequence."""
        _, _, ll = self._forward(obs)
        return ll / len(obs)

# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_digit_dhmm(sequence, dhmm_models):
    best_digit = None
    max_ll     = -np.inf
    for digit, hmm in dhmm_models.items():
        ll = hmm.score(sequence)
        if ll > max_ll:
            max_ll     = ll
            best_digit = digit
    return best_digit

# ---------------------------------------------------------------------------
# Evaluate helper
# ---------------------------------------------------------------------------

def evaluate_hmm(data_vq, dhmm_models):
    true_labels = []
    pred_labels = []
    for item in data_vq:
        true_labels.append(item["digit"])
        pred_labels.append(predict_digit_dhmm(item["sequence"], dhmm_models))

    num_classes = 10
    conf_matrix = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t][p] += 1

    correct  = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / len(true_labels)
    return accuracy, conf_matrix

# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_hmms(trained_dhmms, kmeans, mean, std, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(
            {"dhmms": trained_dhmms, "kmeans": kmeans, "mean": mean, "std": std}, f
        )
    print(f"[HMM] Models saved to {path}")


def load_hmms(path=MODEL_PATH):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["dhmms"], bundle["kmeans"], bundle["mean"], bundle["std"]

# ---------------------------------------------------------------------------
# Main – train & evaluate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading dataset …")
    train_data, val_data, test_data, mean, std = prepare_dataset(RECORDINGS_FOLDER)
    print(f"  Training samples  : {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples      : {len(test_data)}")

    # Build codebook
    all_train_frames = np.vstack([item["features"] for item in train_data])
    print(f"Fitting K-Means ({N_CODEWORDS} codewords) on {all_train_frames.shape[0]} frames …")
    kmeans = KMeans(n_clusters=N_CODEWORDS, max_iter=100, random_state=42)
    kmeans.fit(all_train_frames)
    print("Codebook ready!")

    # Quantise datasets
    train_vq = quantize_dataset(train_data, kmeans)
    val_vq   = quantize_dataset(val_data,   kmeans)
    test_vq  = quantize_dataset(test_data,  kmeans)

    # Group sequences per digit
    hmm_train_seqs = {d: [] for d in range(10)}
    for item in train_vq:
        hmm_train_seqs[item["digit"]].append(item["sequence"])

    # Train
    trained_dhmms = {}
    print("--- TRAINING DISCRETE HMMs ---")
    for digit, seqs in hmm_train_seqs.items():
        print(f"Training Discrete HMM for digit {digit} …")
        hmm = DiscreteHMM(
            n_states=8, n_codewords=N_CODEWORDS, max_iter=100, tol=1e-4, random_state=42
        )
        hmm.fit(seqs)
        trained_dhmms[digit] = hmm

    print("\nAll 10 Discrete HMMs trained!")

    # Evaluate
    val_acc,  val_cm  = evaluate_hmm(val_vq,  trained_dhmms)
    test_acc, test_cm = evaluate_hmm(test_vq, trained_dhmms)

    print("\n===== DISCRETE HMM RESULTS =====")
    print(f"Validation Accuracy : {val_acc  * 100:.2f}%")
    print(f"Test Accuracy       : {test_acc * 100:.2f}%")

    print("\nPer-Digit Validation Accuracy:")
    for digit in range(10):
        total = sum(val_cm[digit])
        if total > 0:
            print(f"  Digit {digit}: {val_cm[digit][digit] / total * 100:.2f}%")

    print("\nPer-Digit Test Accuracy:")
    for digit in range(10):
        total = sum(test_cm[digit])
        if total > 0:
            print(f"  Digit {digit}: {test_cm[digit][digit] / total * 100:.2f}%")

    # Save
    save_hmms(trained_dhmms, kmeans, mean, std)
