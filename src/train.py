"""
Training loop, checkpointing, and a terminal entry point.

Usage:
    python -m src.train --window 500 --epochs 20 --lr 1e-3 --ckpt checkpoints/cnn.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DataConfig, build_dataset, split_by_chromosome
from .model import Creb1CNN


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    seed: int = 0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Convert numpy arrays -> PyTorch DataLoaders."""
    g = torch.Generator().manual_seed(seed)

    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).float()
    Xva = torch.from_numpy(X_val).float()
    yva = torch.from_numpy(y_val).float()

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size, shuffle=True, generator=g,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        TensorDataset(Xva, yva),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        loss = criterion(model(X), y)
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(n, 1)


def save_checkpoint(model: nn.Module, path: Path, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path: Path, model: nn.Module | None = None) -> dict:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["state_dict"])
    return ckpt


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    ckpt_path: Path | None = None,
    log_every: int = 1,
) -> dict:
    """
    Train `model` and return {"train_loss": [...], "val_loss": [...]}.

    Best-val-loss checkpoint is saved if `ckpt_path` is given.
    """
    device = torch.device(device)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate_loss(model, val_loader, criterion, device)

        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:3d}  train {tr:.4f}  val {va:.4f}")

        if ckpt_path is not None and va < best_val:
            best_val = va
            save_checkpoint(model, ckpt_path, epoch=epoch, val_loss=va)

    return history


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train Creb1CNN on cached dataset.")
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--neg-per-pos", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/cnn.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    cfg = DataConfig(window=args.window, neg_per_pos=args.neg_per_pos, seed=args.seed)
    data = build_dataset(cfg)

    # We use the same chromosome split as the notebook, but split *train*
    # further into train/val via held-out chromosomes is NOT what we want —
    # instead, use a random 10% of train chroms' rows as validation here.
    # (The notebook can do a proper 3-way chrom split if desired.)
    
    split = split_by_chromosome(data, test_chroms=cfg.test_chroms)
    X_tr_full, y_tr_full = split["X_train"], split["y_train"]

    rng = np.random.default_rng(args.seed)
    val_idx = rng.random(len(X_tr_full)) < 0.1
    X_val, y_val = X_tr_full[val_idx], y_tr_full[val_idx]
    X_tr, y_tr = X_tr_full[~val_idx], y_tr_full[~val_idx]

    train_loader, val_loader = make_loaders(
        X_tr, y_tr, X_val, y_val,
        batch_size=args.batch_size, seed=args.seed,
    )

    model = Creb1CNN()
    train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        device=args.device, ckpt_path=args.ckpt,
    )
    print(f"checkpoint -> {args.ckpt}")


if __name__ == "__main__":
    main()