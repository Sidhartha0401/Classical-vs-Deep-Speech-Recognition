"""
train.py
--------
Training loop for the CNN-LSTM spoken-digit classifier.

Features
~~~~~~~~
•  Modular logging (Python ``logging`` + console/file handlers)
•  Validation accuracy checked after every epoch
•  Model checkpointing – best val-accuracy checkpoint saved to ``checkpoints/``
•  Early stopping with configurable patience
•  ``ReduceLROnPlateau`` learning-rate scheduler
•  Loss & accuracy curves saved to ``results/plots/``
•  Full reproducibility via ``--seed``

Usage
-----
    python src/train.py --data_dir /path/to/recordings --epochs 60 --batch_size 32

Author : Phase-2 CNN-LSTM project
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Allow running from project root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import build_dataloaders
from model import CNNLSTM
from preprocess import MFCCExtractor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, level: int = logging.INFO) -> None:
    """Configure root logger with console + rotating-file handlers."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor a metric and raise a flag when no improvement is seen
    for ``patience`` consecutive epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = None
        self.triggered = False

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            (self.best is None)
            or (self.mode == "max" and metric > self.best + self.min_delta)
            or (self.mode == "min" and metric < self.best - self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    is_train:  bool,
) -> Tuple[float, float]:
    """
    Run one full pass over ``loader``.

    Returns
    -------
    avg_loss : float
    accuracy : float  (0-1)
    """
    model.train(is_train)
    total_loss = 0.0
    correct    = 0
    total      = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            logits = model(batch_x)
            loss   = criterion(logits, batch_y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds      = logits.argmax(dim=1)
            correct   += (preds == batch_y).sum().item()
            total     += batch_x.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_training_curves(
    history: Dict[str, List],
    save_dir: str,
) -> None:
    """Save loss and accuracy curves to ``save_dir``."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss",  color="#4C72B0", lw=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",    color="#DD8452", lw=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title("Training & Validation Loss", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # --- Accuracy ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], label="Train Accuracy", color="#4C72B0", lw=2)
    ax.plot(epochs, history["val_acc"],   label="Val Accuracy",   color="#DD8452", lw=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Training & Validation Accuracy", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)

    # --- Learning rate ---
    if "lr" in history and len(history["lr"]) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history["lr"], color="#55A868", lw=2)
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("Learning Rate", fontsize=13)
        ax.set_title("Learning Rate Schedule", fontsize=15, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "lr_schedule.png"), dpi=150)
        plt.close(fig)

    logger.info("Training curves saved to '%s'.", save_dir)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ── Reproducibility ──────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    logger.info("Using device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────
    extractor = MFCCExtractor(
        n_mfcc=args.n_mfcc,
        max_len=args.max_len,
        use_delta=True,
    )
    logger.info("Extractor: %s", extractor)

    loaders = build_dataloaders(
        recordings_dir=args.data_dir,
        extractor=extractor,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        cache_features=True,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # ── Model ─────────────────────────────────────────────────────────
    model = CNNLSTM(
        n_channels   = extractor.n_channels,
        n_classes    = 10,
        cnn_channels = (64, 128, 256),
        lstm_hidden  = args.lstm_hidden,
        lstm_layers  = args.lstm_layers,
        lstm_dropout = args.lstm_dropout,
        fc_dropout   = args.fc_dropout,
    ).to(device)
    logger.info("%s", model)

    # ── Loss / Optimiser / Scheduler ─────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=False,
    )
    early_stop = EarlyStopping(patience=args.patience, mode="max")

    # ── Checkpointing ────────────────────────────────────────────────
    ckpt_dir  = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pth"

    # ── History ──────────────────────────────────────────────────────
    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [],
    }
    best_val_acc = 0.0

    # ── Training loop ────────────────────────────────────────────────
    logger.info(
        "Starting training for up to %d epochs (patience=%d) …",
        args.epochs, args.patience
    )
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False
        )

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        ep_time = time.time() - ep_t0
        logger.info(
            "Epoch %3d/%d  |  "
            "train loss: %.4f  acc: %.4f  |  "
            "val   loss: %.4f  acc: %.4f  |  "
            "lr: %.2e  |  %.1fs",
            epoch, args.epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            current_lr, ep_time,
        )

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_acc":     val_acc,
                    "args":        vars(args),
                    "extractor":   {
                        "n_mfcc":    extractor.n_mfcc,
                        "max_len":   extractor.max_len,
                        "use_delta": extractor.use_delta,
                        "n_channels": extractor.n_channels,
                    },
                },
                best_path,
            )
            logger.info("  ✓ New best checkpoint saved (val_acc=%.4f)", val_acc)

        # Early stopping
        if early_stop.step(val_acc):
            logger.info(
                "Early stopping triggered at epoch %d (no improvement for %d epochs).",
                epoch, args.patience,
            )
            break

    total_time = time.time() - t0
    logger.info(
        "Training complete in %.1f s  |  best val accuracy: %.4f",
        total_time, best_val_acc
    )

    # ── Save history ─────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Curves ───────────────────────────────────────────────────────
    save_training_curves(history, str(results_dir / "plots"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the CNN-LSTM spoken-digit classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data_dir",    required=True,
                   help="Directory containing all WAV recordings")
    p.add_argument("--n_mfcc",      type=int,   default=40)
    p.add_argument("--max_len",     type=int,   default=128,
                   help="Fixed number of MFCC time frames")
    p.add_argument("--val_fraction",type=float, default=0.20)
    p.add_argument("--num_workers", type=int,   default=0)

    # Model
    p.add_argument("--lstm_hidden", type=int,   default=256)
    p.add_argument("--lstm_layers", type=int,   default=2)
    p.add_argument("--lstm_dropout",type=float, default=0.30)
    p.add_argument("--fc_dropout",  type=float, default=0.40)

    # Training
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=12)
    p.add_argument("--seed",        type=int,   default=42)

    # Paths
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--results_dir",    default="results")
    p.add_argument("--log_dir",        default="logs")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_dir)
    train(args)
