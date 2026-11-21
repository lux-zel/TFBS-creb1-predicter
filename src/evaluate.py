"""
Metrics, per-chromosome breakdown, and a CLI to score a checkpoint.

Usage:
    python -m src.evaluate --ckpt checkpoints/cnn.pt --window 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from .data import DataConfig, build_dataset, split_by_chromosome
from .model import Creb1CNN
from .train import load_checkpoint


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_prob). y_prob is sigmoid(logits) — the raw model output
    is logits, so we apply sigmoid *here and only here*.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    ys, ps = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.sigmoid(logits)
        ys.append(y.cpu().numpy())
        ps.append(probs.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }


def curves(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """ROC and PR curve points — for plotting in the notebook."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}


def evaluate_by_chromosome(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    chroms: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """One metrics dict per chromosome. Useful to spot a single bad chrom."""
    out: dict[str, dict] = {}
    for c in np.unique(chroms):
        mask = chroms == c
        yt, yp = y_true[mask], y_prob[mask]
        if len(np.unique(yt)) < 2:
            # AUROC needs both classes
            out[str(c)] = {
                "n": int(len(yt)),
                "positives": int(yt.sum()),
                "note": "single-class; AUROC undefined",
            }
            continue
        out[str(c)] = compute_metrics(yt, yp, threshold)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a Creb1CNN checkpoint.")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--neg-per-pos", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None,
                   help="If set, write metrics JSON here.")
    args = p.parse_args(argv)

    cfg = DataConfig(window=args.window, neg_per_pos=args.neg_per_pos, seed=args.seed)
    data = build_dataset(cfg)
    split = split_by_chromosome(data, test_chroms=cfg.test_chroms)

    X_te, y_te = split["X_test"], split["y_test"]
    chroms_te = split["chroms_test"]

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).float()),
        batch_size=args.batch_size, shuffle=False,
    )

    model = Creb1CNN()
    load_checkpoint(args.ckpt, model)

    y_true, y_prob = predict(model, loader, device=args.device)
    overall = compute_metrics(y_true, y_prob, args.threshold)
    per_chrom = evaluate_by_chromosome(y_true, y_prob, chroms_te, args.threshold)

    print("=== overall (held-out chroms) ===")
    for k, v in overall.items():
        print(f"  {k:10s} {v}")

    print("\n=== per chromosome ===")
    for c, m in per_chrom.items():
        if "auroc" in m:
            print(f"  chr{c:<3s} n={m['n']:6d}  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}")
        else:
            print(f"  chr{c:<3s} {m}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"overall": overall, "per_chromosome": per_chrom}, indent=2
        ))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()