"""
data_loader.py
--------------
Custom PyTorch Dataset for the spoken-digit recognition task.

Split strategy (hard-coded per project specification):
    • Speakers 1-5  →  training / validation pool
    • Speaker  6    →  held-out test set

File naming convention:
    {digit}_speaker{speaker_id}_{index}.wav
    e.g.  4_speaker2_17.wav

Author : Phase-2 CNN-LSTM project
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from preprocess import MFCCExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(r"^(\d+)_speaker(\d+)_(\d+)\.wav$", re.IGNORECASE)


def parse_filename(filename: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse a wav filename into (digit, speaker_id, index).

    Returns ``None`` if the filename does not match the expected pattern.
    """
    m = _FNAME_RE.match(os.path.basename(filename))
    if m is None:
        return None
    digit, speaker, idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return digit, speaker, idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpokenDigitDataset(Dataset):
    """
    PyTorch Dataset for spoken digit WAV files.

    Parameters
    ----------
    file_paths : list of str
        Absolute paths to the WAV files to include.
    labels : list of int
        Digit labels (0-9) aligned with ``file_paths``.
    extractor : MFCCExtractor
        Pre-built extractor; used to compute/cache features lazily.
    cache_features : bool
        If True, extract and hold all MFCCs in RAM at construction time.
        Recommended when the dataset fits in memory (≤ a few GB).
    """

    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        extractor: "MFCCExtractor",
        cache_features: bool = True,
    ):
        assert len(file_paths) == len(labels), "Mismatched lengths"
        self.file_paths = file_paths
        self.labels = labels
        self.extractor = extractor
        self._cache: Optional[List[torch.Tensor]] = None

        if cache_features:
            logger.info("Pre-extracting and caching MFCC features …")
            self._cache = [
                self.extractor.extract(p) for p in self.file_paths
            ]
            logger.info("Feature cache built  (%d samples).", len(self._cache))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self._cache is not None:
            features = self._cache[idx]
        else:
            features = self.extractor.extract(self.file_paths[idx])
        return features, self.labels[idx]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _collect_files(recordings_dir: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Walk ``recordings_dir`` and collect paths, labels, and speaker IDs
    for all valid WAV files.
    """
    paths, labels, speakers = [], [], []
    recordings_dir = Path(recordings_dir)

    if not recordings_dir.exists():
        raise FileNotFoundError(
            f"Recordings directory not found: {recordings_dir}\n"
            "Set the correct path via --data_dir or in configs/config.yaml"
        )

    for fpath in sorted(recordings_dir.iterdir()):
        if fpath.suffix.lower() != ".wav":
            continue
        parsed = parse_filename(fpath.name)
        if parsed is None:
            logger.warning("Skipping unrecognised filename: %s", fpath.name)
            continue
        digit, speaker, _ = parsed
        paths.append(str(fpath))
        labels.append(digit)
        speakers.append(speaker)

    logger.info(
        "Discovered %d WAV files in '%s'.", len(paths), recordings_dir
    )
    return paths, labels, speakers


def build_dataloaders(
    recordings_dir: str,
    extractor: "MFCCExtractor",
    val_fraction: float = 0.20,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    cache_features: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / validation / test DataLoaders.

    Split strategy
    ~~~~~~~~~~~~~~
    •  Speakers 1-5  →  pool is split 80 / 20  (train / val)
    •  Speaker  6    →  test set

    Parameters
    ----------
    recordings_dir : str
        Root directory containing all WAV files (flat layout).
    extractor : MFCCExtractor
        Configured extractor instance.
    val_fraction : float
        Fraction of the speaker-1-5 pool reserved for validation.
    batch_size : int
    num_workers : int
    seed : int
        Random seed for reproducible val split.
    cache_features : bool
        Pre-cache all MFCC tensors in RAM.

    Returns
    -------
    dict with keys "train", "val", "test" → DataLoader instances.
    """
    paths, labels, speakers = _collect_files(recordings_dir)

    # --- Speaker-based split -----------------------------------------------
    train_val_paths  = [p for p, s in zip(paths, speakers) if 1 <= s <= 5]
    train_val_labels = [l for l, s in zip(labels, speakers) if 1 <= s <= 5]
    test_paths       = [p for p, s in zip(paths, speakers) if s == 6]
    test_labels      = [l for l, s in zip(labels, speakers) if s == 6]

    logger.info(
        "Speaker split → train/val pool: %d  |  test (spk-6): %d",
        len(train_val_paths), len(test_paths),
    )

    if not train_val_paths:
        raise RuntimeError("No training/validation files found (speakers 1-5).")
    if not test_paths:
        raise RuntimeError("No test files found (speaker 6).")

    # Build full train+val dataset first (so we can share the feature cache)
    full_tv_dataset = SpokenDigitDataset(
        train_val_paths, train_val_labels, extractor, cache_features=cache_features
    )

    # --- Val split (random, but reproducible) ------------------------------
    n_val   = int(len(full_tv_dataset) * val_fraction)
    n_train = len(full_tv_dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_tv_dataset, [n_train, n_val], generator=generator
    )

    # Test dataset (always fully cached for speed during evaluation)
    test_dataset = SpokenDigitDataset(
        test_paths, test_labels, extractor, cache_features=cache_features
    )

    logger.info(
        "Dataset sizes → train: %d | val: %d | test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train": DataLoader(train_dataset, shuffle=True,  **loader_kwargs),
        "val":   DataLoader(val_dataset,   shuffle=False, **loader_kwargs),
        "test":  DataLoader(test_dataset,  shuffle=False, **loader_kwargs),
    }




if __name__ == "__main__":
    import argparse
    from preprocess import MFCCExtractor

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    ext = MFCCExtractor()
    loaders = build_dataloaders(args.data_dir, ext, cache_features=False)
    for split, dl in loaders.items():
        batch = next(iter(dl))
        print(f"[{split}]  batch shape: {batch[0].shape}  label: {batch[1]}")
