"""
gmm.py
------
Gaussian Mixture Model (GMM) for spoken-digit recognition.

Usage (training / evaluation):
    python gmm.py

The script expects the recordings folder to exist at the path set in
RECORDINGS_FOLDER.  It trains one GMM per digit (0-9), evaluates on the
validation and test splits, prints per-digit accuracy, and saves the trained
models to `trained_gmms.pkl` in the same directory.

Exported symbols used by inference.py:
    GaussianMixtureModel   – the GMM class
    predict_digit_gmm      – classification helper
    load_gmms              – loads a previously saved model bundle
    FEATURE_FUNCTIONS      – (extract_features, extract_for_inference)
"""

import os
import pickle
import random

import librosa
import numpy as np

# ---------------------------------------------------------------------------
# Path to your recordings dataset (edit as needed)
# ---------------------------------------------------------------------------
RECORDINGS_FOLDER = "/Users/sidharthadurgam/Downloads/recordings"
MODEL_PATH        = os.path.join(os.path.dirname(__file__), "trained_gmms.pkl")

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def extract_features(audio_path, n_mfcc=20):
    signal, sr = librosa.load(audio_path, sr=None)
    signal = pre_emphasis(signal)

    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=160
    )

    T     = mfcc.shape[1]
    width = min(9, T if T % 2 == 1 else T - 1)
    if width < 3:
        width = 3

    delta  = librosa.feature.delta(mfcc, width=width)
    delta2 = librosa.feature.delta(mfcc, order=2, width=width)

    features = np.vstack((mfcc, delta, delta2))
    return features.T          # shape (T, 60)


def extract_for_inference(audio_file, mean, std):
    features = extract_features(audio_file)
    return (features - mean) / std

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def parse_filename(filename):
    parts  = filename.replace(".wav", "").split("_")
    digit  = int(parts[0])
    speaker = int(parts[1].replace("speaker", ""))
    return digit, speaker


def load_dataset(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            path          = os.path.join(folder_path, file)
            digit, speaker = parse_filename(file)
            features      = extract_features(path)
            data.append({"features": features, "digit": digit, "speaker": speaker})
    return data


def split_data(data):
    speaker_1_to_5 = [item for item in data if item["speaker"] <= 5]
    test           = [item for item in data if item["speaker"] == 6]

    random.seed(42)
    random.shuffle(speaker_1_to_5)

    split_idx = int(0.8 * len(speaker_1_to_5))
    train     = speaker_1_to_5[:split_idx]
    val       = speaker_1_to_5[split_idx:]
    return train, val, test


def compute_normalization(train_data):
    all_features = np.vstack([item["features"] for item in train_data])
    mean = np.mean(all_features, axis=0)
    std  = np.std(all_features, axis=0) + 1e-8
    return mean, std


def normalize(data, mean, std):
    for item in data:
        item["features"] = (item["features"] - mean) / std
    return data


def prepare_dataset(folder_path):
    data              = load_dataset(folder_path)
    train, val, test  = split_data(data)
    mean, std         = compute_normalization(train)
    train = normalize(train, mean, std)
    val   = normalize(val,   mean, std)
    test  = normalize(test,  mean, std)
    return train, val, test, mean, std

# ---------------------------------------------------------------------------
# Gaussian Mixture Model
# ---------------------------------------------------------------------------

class GaussianMixtureModel:
    """Diagonal-covariance GMM trained with EM."""

    def __init__(self, n_components=2, max_iter=100, tol=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state

        self.pi  = None
        self.mu  = None
        self.sigma = None
        self.log_likelihood_history = []

    # ------------------------------------------------------------------
    def _ensure_2d(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _initialize_parameters(self, X):
        X = self._ensure_2d(X)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)

        self.pi    = np.ones(self.n_components) / self.n_components
        random_idx = rng.choice(n_samples, self.n_components, replace=False)
        self.mu    = X[random_idx]
        self.sigma = np.ones((self.n_components, n_features)) + 1e-6

    def _log_gaussian_pdf(self, X, mu, sigma):
        n_features = X.shape[1]
        log_const  = -0.5 * n_features * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma))
        diff       = X - mu
        log_exp    = -0.5 * np.sum((diff ** 2) / sigma, axis=1)
        return log_const + log_exp

    def _e_step(self, X):
        n_samples          = X.shape[0]
        log_weighted_probs = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            log_weighted_probs[:, k] = (
                np.log(self.pi[k]) + self._log_gaussian_pdf(X, self.mu[k], self.sigma[k])
            )

        max_vals = np.max(log_weighted_probs, axis=1, keepdims=True)
        log_sum  = max_vals + np.log(
            np.sum(np.exp(log_weighted_probs - max_vals), axis=1, keepdims=True)
        )
        log_resp = log_weighted_probs - log_sum
        return np.exp(log_resp)

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        N_k = np.sum(responsibilities, axis=0)

        self.pi    = N_k / n_samples
        self.mu    = np.zeros((self.n_components, n_features))
        self.sigma = np.zeros((self.n_components, n_features))

        for k in range(self.n_components):
            gamma_k    = responsibilities[:, k]
            self.mu[k] = np.sum(gamma_k.reshape(-1, 1) * X, axis=0) / N_k[k]
            diff       = X - self.mu[k]
            self.sigma[k] = (
                np.sum(gamma_k.reshape(-1, 1) * (diff ** 2), axis=0) / N_k[k] + 1e-6
            )

    def _compute_log_likelihood(self, X):
        n_samples          = X.shape[0]
        log_weighted_probs = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            log_weighted_probs[:, k] = (
                np.log(self.pi[k]) + self._log_gaussian_pdf(X, self.mu[k], self.sigma[k])
            )

        max_vals = np.max(log_weighted_probs, axis=1, keepdims=True)
        log_sum  = max_vals + np.log(
            np.sum(np.exp(log_weighted_probs - max_vals), axis=1, keepdims=True)
        )
        return np.sum(log_sum)

    # ------------------------------------------------------------------
    def fit(self, X):
        X = self._ensure_2d(X)
        self._initialize_parameters(X)

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            curr_ll = self._compute_log_likelihood(X)
            self.log_likelihood_history.append(curr_ll)

            imp = curr_ll - prev_ll
            if 0 <= imp < self.tol:
                break
            prev_ll = curr_ll
        return self

    def predict_proba(self, X):
        X = self._ensure_2d(X)
        return self._e_step(X)

    def predict(self, X):
        X = self._ensure_2d(X)
        return np.argmax(self._e_step(X), axis=1)

    # used by inference.py to score a sequence
    def score(self, X):
        X = self._ensure_2d(X)
        return self._compute_log_likelihood(X)

# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_digit_gmm(test_features, gmm_models):
    """Return the digit whose GMM gives the highest log-likelihood."""
    best_digit = None
    max_ll     = -np.inf
    for digit, gmm in gmm_models.items():
        s = gmm._compute_log_likelihood(test_features)
        if s > max_ll:
            max_ll     = s
            best_digit = digit
    return best_digit

# ---------------------------------------------------------------------------
# Evaluate helper (mirrors notebook)
# ---------------------------------------------------------------------------

def evaluate_gmm(data, gmm_models):
    true_labels = []
    pred_labels = []
    for item in data:
        true_labels.append(item["digit"])
        pred_labels.append(predict_digit_gmm(item["features"], gmm_models))

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

def save_gmms(trained_gmms, mean, std, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump({"gmms": trained_gmms, "mean": mean, "std": std}, f)
    print(f"[GMM] Models saved to {path}")


def load_gmms(path=MODEL_PATH):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["gmms"], bundle["mean"], bundle["std"]

# ---------------------------------------------------------------------------
# Main – train & evaluate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading dataset …")
    train_data, val_data, test_data, mean, std = prepare_dataset(RECORDINGS_FOLDER)
    print(f"  Training samples : {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples     : {len(test_data)}")

    # Group training frames per digit
    raw_vars = {d: [] for d in range(10)}
    for item in train_data:
        raw_vars[item["digit"]].append(item["features"])

    clean_mats = {}
    for digit, seqs in raw_vars.items():
        mat = np.array(np.vstack(seqs), dtype=np.float32)
        clean_mats[digit] = mat
        print(f"  Digit {digit}: {mat.shape}")

    # Train
    trained_gmms = {}
    for digit, X_train in clean_mats.items():
        print(f"Training GMM for digit {digit} …")
        gmm = GaussianMixtureModel(n_components=8, max_iter=100, tol=1e-6, random_state=42)
        gmm.fit(X_train)
        trained_gmms[digit] = gmm

    # Evaluate
    val_acc,  val_cm  = evaluate_gmm(val_data,  trained_gmms)
    test_acc, test_cm = evaluate_gmm(test_data, trained_gmms)

    print("\n===== GMM RESULTS =====")
    print(f"Validation Accuracy : {val_acc  * 100:.2f}%")
    print(f"Test Accuracy       : {test_acc * 100:.2f}%")

    print("\nGMM Per-Digit Validation Accuracy:")
    for digit in range(10):
        total = sum(val_cm[digit])
        if total > 0:
            print(f"  Digit {digit}: {val_cm[digit][digit] / total * 100:.2f}%")

    print("\nGMM Per-Digit Test Accuracy:")
    for digit in range(10):
        total = sum(test_cm[digit])
        if total > 0:
            print(f"  Digit {digit}: {test_cm[digit][digit] / total * 100:.2f}%")

    # Save models
    save_gmms(trained_gmms, mean, std)
