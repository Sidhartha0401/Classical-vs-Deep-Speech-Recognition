"""
preprocess.py
-------------
Audio preprocessing pipeline:
    1. Load WAV file with librosa.
    2. Apply pre-emphasis filter (high-frequency boost).
    3. Extract n_mfcc Mel-Frequency Cepstral Coefficients (MFCCs).
    4. Optionally append Δ and ΔΔ coefficients (delta features).
    5. Pad or truncate the time-dimension to a fixed length.
    6. Return a (C, T) float32 tensor ready for the CNN-LSTM model.

The MFCCExtractor class is stateless and safe to share between workers.

Author : Phase-2 CNN-LSTM project
"""

import logging
from typing import Optional

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_N_MFCC     = 40       # number of MFCC coefficients
DEFAULT_MAX_LEN    = 128      # fixed time-axis length (frames)
DEFAULT_SAMPLE_RATE = None    # None → keep native sample rate
DEFAULT_N_FFT      = 512
DEFAULT_HOP_LENGTH = 160      # ~10 ms at 16 kHz
DEFAULT_PRE_ALPHA  = 0.97     # pre-emphasis coefficient


# ---------------------------------------------------------------------------
# MFCCExtractor
# ---------------------------------------------------------------------------

class MFCCExtractor:
    """
    Stateless MFCC feature extractor.

    Parameters
    ----------
    n_mfcc : int
        Number of MFCC coefficients (default 40).
    max_len : int
        Fixed time length in frames.  Shorter utterances are zero-padded;
        longer ones are truncated (centre-crop).
    sample_rate : int or None
        Target sample rate.  ``None`` keeps the file's native rate.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop size between frames.
    pre_emphasis : float
        Pre-emphasis filter coefficient.  Set to 0 to disable.
    use_delta : bool
        If True, append Δ and ΔΔ features, tripling the channel count
        (n_mfcc → 3 * n_mfcc).
    """

    def __init__(
        self,
        n_mfcc:      int   = DEFAULT_N_MFCC,
        max_len:     int   = DEFAULT_MAX_LEN,
        sample_rate: Optional[int] = DEFAULT_SAMPLE_RATE,
        n_fft:       int   = DEFAULT_N_FFT,
        hop_length:  int   = DEFAULT_HOP_LENGTH,
        pre_emphasis: float = DEFAULT_PRE_ALPHA,
        use_delta:   bool  = True,
    ):
        self.n_mfcc       = n_mfcc
        self.max_len      = max_len
        self.sample_rate  = sample_rate
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.pre_emphasis = pre_emphasis
        self.use_delta    = use_delta

    # ------------------------------------------------------------------
    @property
    def n_channels(self) -> int:
        """Number of feature channels in the output tensor."""
        return self.n_mfcc * (3 if self.use_delta else 1)

    # ------------------------------------------------------------------
    def _apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        if self.pre_emphasis > 0:
            return np.append(signal[0], signal[1:] - self.pre_emphasis * signal[:-1])
        return signal

    def _pad_or_truncate(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Ensure MFCC has exactly ``max_len`` time frames.

        Pads with zeros on the right; centre-crops if too long.
        Shape in:  (C, T)
        Shape out: (C, max_len)
        """
        C, T = mfcc.shape
        if T == self.max_len:
            return mfcc
        if T < self.max_len:
            pad = np.zeros((C, self.max_len - T), dtype=mfcc.dtype)
            return np.concatenate([mfcc, pad], axis=1)
        # Centre-crop
        start = (T - self.max_len) // 2
        return mfcc[:, start : start + self.max_len]

    def extract(self, audio_path: str) -> torch.Tensor:
        """
        Load an audio file and return a float32 tensor of shape
        ``(n_channels, max_len)``.
        """
        try:
            signal, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load '{audio_path}': {exc}") from exc

        # 1. Pre-emphasis
        signal = self._apply_pre_emphasis(signal)

        # 2. MFCC
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )                                        # (n_mfcc, T)

        # 3. Delta features
        if self.use_delta:
            T     = mfcc.shape[1]
            width = min(9, T if T % 2 == 1 else T - 1)
            width = max(width, 3)

            delta  = librosa.feature.delta(mfcc, width=width)
            delta2 = librosa.feature.delta(mfcc, order=2, width=width)
            mfcc   = np.vstack([mfcc, delta, delta2])  # (3*n_mfcc, T)

        # 4. Pad / truncate
        mfcc = self._pad_or_truncate(mfcc)       # (C, max_len)

        # 5. Per-channel normalisation (zero-mean, unit-variance)
        mean = mfcc.mean(axis=1, keepdims=True)
        std  = mfcc.std(axis=1, keepdims=True) + 1e-8
        mfcc = (mfcc - mean) / std

        return torch.tensor(mfcc, dtype=torch.float32)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"MFCCExtractor("
            f"n_mfcc={self.n_mfcc}, "
            f"max_len={self.max_len}, "
            f"use_delta={self.use_delta}, "
            f"n_channels={self.n_channels})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path/to/file.wav>")
        sys.exit(1)

    logging.basicConfig(level=logging.DEBUG)
    ext = MFCCExtractor()
    t   = ext.extract(sys.argv[1])
    print(f"Extractor : {ext}")
    print(f"Output    : shape={t.shape}  dtype={t.dtype}  "
          f"min={t.min():.3f}  max={t.max():.3f}")
