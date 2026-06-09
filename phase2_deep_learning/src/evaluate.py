"""
evaluate.py
-----------
Load a trained CNN-LSTM checkpoint and produce a full evaluation report:

    •  Validation accuracy
    •  Test accuracy (Speaker 6 – held-out)
    •  Per-digit accuracy table (train/val pool vs. test)
    •  Confusion matrices  (saved as PNG)
    •  Classification report (precision / recall / F1)
    •  Results JSON saved to ``results/``

Usage
-----
    python src/evaluate.py \\
        --data_dir /path/to/recordings \\
        --checkpoint checkpoints/best_model.pth

Author : Phase-2 CNN-LSTM project
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import build_dataloaders
from model import CNNLSTM
from preprocess import MFCCExtractor

logger = logging.getLogger(__name__)

DIGIT_NAMES = [str(d) for d in range(10)]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> Tuple[List, List]:
    """Run inference and return (true_labels, pred_labels)."""
    model.eval()
    y_true, y_pred = [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        logits  = model(batch_x)
        preds   = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch_y.tolist())
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true:    List[int],
    y_pred:    List[int],
    title:     str,
    save_path: str,
    normalise: bool = True,
) -> None:
    """
    Render and save a confusion matrix.

    Parameters
    ----------
    normalise : bool
        If True, normalise by row (true-label frequency).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        cm_plot  = np.where(row_sums == 0, 0, cm / (row_sums + 1e-8))
        fmt, vmin, vmax = ".2f", 0.0, 1.0
        cbar_label = "Recall (row-normalised)"
    else:
        cm_plot  = cm
        fmt, vmin, vmax = "d", 0, None
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_plot,
        annot=True, fmt=fmt, cmap="Blues",
        xticklabels=DIGIT_NAMES, yticklabels=DIGIT_NAMES,
        linewidths=0.5, linecolor="lightgrey",
        vmin=vmin, vmax=vmax,
        ax=ax,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
    )
    ax.set_xlabel("Predicted Digit", fontsize=13, labelpad=8)
    ax.set_ylabel("True Digit",      fontsize=13, labelpad=8)
    ax.set_title(title,              fontsize=15, fontweight="bold", pad=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)


def plot_per_digit_accuracy(
    val_acc_per_digit:  Dict[int, float],
    test_acc_per_digit: Dict[int, float],
    save_path: str,
) -> None:
    """Bar chart comparing per-digit accuracy on val vs. test split."""
    digits = list(range(10))
    val_vals  = [val_acc_per_digit.get(d, 0)  for d in digits]
    test_vals = [test_acc_per_digit.get(d, 0) for d in digits]

    x     = np.arange(len(digits))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width / 2, val_vals,  width, label="Validation (Spk 1-5)", color="#4C72B0", alpha=0.9)
    bars2 = ax.bar(x + width / 2, test_vals, width, label="Test (Speaker 6)",     color="#DD8452", alpha=0.9)

    ax.set_xlabel("Digit", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("CNN-LSTM Per-Digit Accuracy", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(DIGIT_NAMES)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#4C72B0")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#DD8452")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Per-digit accuracy chart saved → %s", save_path)


# ---------------------------------------------------------------------------
# Per-digit accuracy helper
# ---------------------------------------------------------------------------

def per_digit_accuracy(y_true: List[int], y_pred: List[int]) -> Dict[int, float]:
    correct = {d: 0 for d in range(10)}
    total   = {d: 0 for d in range(10)}
    for t, p in zip(y_true, y_pred):
        total[t] += 1
        if t == p:
            correct[t] += 1
    return {d: correct[d] / total[d] if total[d] > 0 else 0.0 for d in range(10)}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    # ── Device ────────────────────────────────────────────────────────
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    logger.info("Using device: %s", device)

    # ── Load checkpoint ───────────────────────────────────────────────
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    ext_cfg   = ckpt["extractor"]
    model_cfg = ckpt.get("args", {})

    extractor = MFCCExtractor(
        n_mfcc=ext_cfg["n_mfcc"],
        max_len=ext_cfg["max_len"],
        use_delta=ext_cfg["use_delta"],
    )

    model = CNNLSTM(
        n_channels   = ext_cfg["n_channels"],
        n_classes    = 10,
        cnn_channels = (64, 128, 256),
        lstm_hidden  = model_cfg.get("lstm_hidden", 256),
        lstm_layers  = model_cfg.get("lstm_layers", 2),
        lstm_dropout = model_cfg.get("lstm_dropout", 0.30),
        fc_dropout   = model_cfg.get("fc_dropout",   0.40),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(
        "Checkpoint loaded (epoch %d, val_acc=%.4f)",
        ckpt.get("epoch", -1), ckpt.get("val_acc", -1)
    )

    # ── DataLoaders ───────────────────────────────────────────────────
    loaders = build_dataloaders(
        recordings_dir=args.data_dir,
        extractor=extractor,
        val_fraction=0.20,
        batch_size=args.batch_size,
        num_workers=0,
        cache_features=True,
    )

    # ── Inference ─────────────────────────────────────────────────────
    results_dir  = Path(args.results_dir)
    plots_dir    = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "val":  ("Validation (Speakers 1-5)", loaders["val"]),
        "test": ("Test (Speaker 6)",           loaders["test"]),
    }

    summary: Dict = {}

    for split, (split_label, loader) in splits.items():
        logger.info("Running inference on %s split …", split)
        y_true, y_pred = predict(model, loader, device)

        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        pda      = per_digit_accuracy(y_true, y_pred)

        logger.info("  %s accuracy: %.4f  (%d/%d)",
                    split_label, accuracy,
                    int(accuracy * len(y_true)), len(y_true))

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=DIGIT_NAMES,
            digits=4,
            zero_division=0,
        )
        print(f"\n{'='*60}")
        print(f"  {split_label.upper()}  —  Accuracy: {accuracy:.4f}")
        print("="*60)
        print(report)

        # Confusion matrices
        plot_confusion_matrix(
            y_true, y_pred,
            title=f"CNN-LSTM Confusion Matrix — {split_label}",
            save_path=str(plots_dir / f"confusion_matrix_{split}.png"),
            normalise=True,
        )

        summary[split] = {
            "accuracy": accuracy,
            "per_digit_accuracy": pda,
            "n_samples": len(y_true),
        }

    # Per-digit comparison chart
    plot_per_digit_accuracy(
        val_acc_per_digit  = summary["val"]["per_digit_accuracy"],
        test_acc_per_digit = summary["test"]["per_digit_accuracy"],
        save_path          = str(plots_dir / "per_digit_accuracy.png"),
    )

    # Final summary to console
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"  Validation accuracy : {summary['val']['accuracy']*100:.2f}%")
    print(f"  Test accuracy       : {summary['test']['accuracy']*100:.2f}%")
    print()
    print("  Per-digit test accuracy:")
    for d in range(10):
        acc = summary["test"]["per_digit_accuracy"][d]
        bar = "█" * int(acc * 20)
        print(f"    Digit {d}:  {acc*100:6.2f}%  {bar}")
    print("="*60)

    # Save JSON
    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Evaluation summary saved → %s", summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the CNN-LSTM spoken-digit classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--checkpoint",  required=True,
                   help="Path to a .pth checkpoint file")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    args = parse_args()
    evaluate(args)
