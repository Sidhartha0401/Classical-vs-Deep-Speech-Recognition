"""
inference.py
------------
Spoken-digit inference: given a WAV recording, returns the predicted digit
using the **best** of the two trained models (GMM and Discrete HMM).

In the notebook experiments:
    GMM  achieved 97.40 % validation accuracy
    HMM  achieved 92.20 % validation accuracy

The GMM is therefore used as the primary predictor, with the HMM available
as a fallback / ensemble option.

Usage (command-line):
    python inference.py <path/to/recording.wav>

    Optional flags:
        --model gmm       use only the GMM
        --model hmm       use only the HMM
        --model best      use the model with the higher log-likelihood on this
                          recording (default)
        --model ensemble  average the (normalised) scores of both models

Programmatic usage:
    from inference import predict_from_file
    digit = predict_from_file("my_recording.wav")   # returns int 0-9
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Try to import the model packages from the same directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gmm import (
    GaussianMixtureModel,
    load_gmms,
    extract_for_inference,
    MODEL_PATH as GMM_MODEL_PATH,
)
from hmm import (
    DiscreteHMM,
    KMeans,
    load_hmms,
    quantize_sequence,
    MODEL_PATH as HMM_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _score_all_gmm(features, trained_gmms):
    """Return dict {digit: log-likelihood} for the GMM ensemble."""
    return {digit: gmm.score(features) for digit, gmm in trained_gmms.items()}


def _score_all_hmm(sequence, trained_dhmms):
    """Return dict {digit: normalised log-likelihood} for the HMM ensemble."""
    return {digit: hmm.score(sequence) for digit, hmm in trained_dhmms.items()}


def predict_from_file(
    audio_path,
    mode="best",
    gmm_bundle=None,
    hmm_bundle=None,
):
    """
    Predict the spoken digit in ``audio_path``.

    Parameters
    ----------
    audio_path : str
        Path to the WAV file.
    mode : str
        One of ``"gmm"``, ``"hmm"``, ``"best"``, or ``"ensemble"``.
        Default is ``"best"``.
    gmm_bundle : tuple or None
        Pre-loaded (trained_gmms, mean, std).  If None the bundle is loaded
        from ``trained_gmms.pkl``.
    hmm_bundle : tuple or None
        Pre-loaded (trained_dhmms, kmeans, mean, std).  If None the bundle is
        loaded from ``trained_hmms.pkl``.

    Returns
    -------
    int  – predicted digit (0-9)
    """
    if mode not in ("gmm", "hmm", "best", "ensemble"):
        raise ValueError(f"Unknown mode '{mode}'. Choose from gmm/hmm/best/ensemble.")

    # ---- Load model bundles (lazy) ----------------------------------------
    if mode in ("gmm", "best", "ensemble"):
        if gmm_bundle is None:
            gmm_bundle = load_gmms(GMM_MODEL_PATH)
        trained_gmms, gmm_mean, gmm_std = gmm_bundle

    if mode in ("hmm", "best", "ensemble"):
        if hmm_bundle is None:
            hmm_bundle = load_hmms(HMM_MODEL_PATH)
        trained_dhmms, kmeans, hmm_mean, hmm_std = hmm_bundle

    # ---- Feature extraction -----------------------------------------------
    if mode in ("gmm", "best", "ensemble"):
        gmm_features = extract_for_inference(audio_path, gmm_mean, gmm_std)
        gmm_scores   = _score_all_gmm(gmm_features, trained_gmms)

    if mode in ("hmm", "best", "ensemble"):
        hmm_features = extract_for_inference(audio_path, hmm_mean, hmm_std)
        hmm_sequence = quantize_sequence(hmm_features, kmeans)
        hmm_scores   = _score_all_hmm(hmm_sequence, trained_dhmms)

    # ---- Decide -----------------------------------------------------------
    if mode == "gmm":
        return max(gmm_scores, key=gmm_scores.get)

    if mode == "hmm":
        return max(hmm_scores, key=hmm_scores.get)

    if mode == "best":
        # Compare the best score from each model (both are log-likelihoods).
        # GMM: total log-likelihood over all frames.
        # HMM: mean log-likelihood per frame.
        # We normalise GMM by number of frames for a fair comparison.
        T = gmm_features.shape[0]
        best_gmm_ll = max(gmm_scores.values()) / (T + 1e-9)
        best_hmm_ll = max(hmm_scores.values())

        if best_gmm_ll >= best_hmm_ll:
            return max(gmm_scores, key=gmm_scores.get)
        else:
            return max(hmm_scores, key=hmm_scores.get)

    if mode == "ensemble":
        # Normalise each model's scores so they are on comparable scales,
        # then average and pick the winner.
        T = gmm_features.shape[0]

        gmm_arr = np.array([gmm_scores[d] for d in range(10)]) / (T + 1e-9)
        hmm_arr = np.array([hmm_scores[d] for d in range(10)])

        # Min-max normalise each to [0, 1] before averaging
        def _norm(arr):
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-12:
                return np.ones_like(arr) / len(arr)
            return (arr - lo) / (hi - lo)

        combined = (_norm(gmm_arr) + _norm(hmm_arr)) / 2.0
        return int(np.argmax(combined))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict the spoken digit (0-9) in a WAV recording."
    )
    parser.add_argument("audio_path", help="Path to the input WAV file.")
    parser.add_argument(
        "--model",
        choices=["gmm", "hmm", "best", "ensemble"],
        default="best",
        help=(
            "Which model to use.\n"
            "  gmm      – use only the GMM (best overall accuracy)\n"
            "  hmm      – use only the Discrete HMM\n"
            "  best     – use whichever model is more confident (default)\n"
            "  ensemble – average normalised scores from both models"
        ),
    )
    parser.add_argument(
        "--gmm-model",
        default=GMM_MODEL_PATH,
        help="Path to the trained_gmms.pkl bundle.",
    )
    parser.add_argument(
        "--hmm-model",
        default=HMM_MODEL_PATH,
        help="Path to the trained_hmms.pkl bundle.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.audio_path):
        print(f"Error: file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    # Pre-load bundles so we can pass custom paths from CLI flags
    gmm_bundle = None
    hmm_bundle = None
    if args.model in ("gmm", "best", "ensemble"):
        gmm_bundle = load_gmms(args.gmm_model)
    if args.model in ("hmm", "best", "ensemble"):
        hmm_bundle = load_hmms(args.hmm_model)

    predicted = predict_from_file(
        args.audio_path,
        mode=args.model,
        gmm_bundle=gmm_bundle,
        hmm_bundle=hmm_bundle,
    )
    print(f"Predicted digit: {predicted}")


if __name__ == "__main__":
    main()
