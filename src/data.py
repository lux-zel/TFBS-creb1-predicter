
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pyfaidx import Fasta

#constants, voor de chromosome identificatie en de one_hot 
#encoding van de basen.
CANONICAL_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]

NARROWPEAK_COLS = [
    "chrom", "start", "end", "name", "score", "strand",
    "signalValue", "pValue", "qValue", "summit"]

BASE_TO_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass
class DataConfig:
    bed_path: Path
    fasta_path: Path
    out_dir: Path
    window: int = 500
    neg_per_pos: int = 1
    seed: int = 0
    test_chroms: tuple[str, ...] = ("1", "2")


def load_peaks(path: Path) -> pd.DataFrame:
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Peak file not found: {path}")

    df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=NARROWPEAK_COLS,
            usecols=["chrom", "start", "end", "signalvalue", "summit"],
            dtype={"chrom": str})

    df["chrom"] = df["chrom"].str.replace(r"^chr", "", regex=True)

    df = df[df["chrom"].isin(CANONICAL_CHROMS)]

    for col in ("start", "end", "summit"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")

    df = df[(df["end"] > df["start"]) & (df["start"] >= 0)]

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No peaks left after filtering to canonical chroms in {path}. "
            f"Check that the BED uses 'chr'-prefixed names."
        )
    return df


def one_hot(seq: str) -> np.ndarray:
    seq = seq.upper()
    out = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        idx = BASE_TO_IDX.get(b)
        if idx is not None:
            out[idx, i] = 1.0
    return out



def extract_window(fa, chrom: str, center: int, width: int) -> np.ndarray:
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    half = width // 2
    start0 = center - half
    end0 = start0 + width

    chrom_len = len(fa[chrom])

    clip_start = max(start0, 0)
    clip_end = min(end0, chrom_len)

    seq = str(fa[chrom][clip_start + 1: clip_end + 1]).upper()

    enc = one_hot(seq)

    if enc.shape[1] != width:
        pad_left = clip_start - start0

        pad_right = end0 - clip_end
        enc = np.pad(enc, ((0, 0), (pad_left, pad_right)), constant_values = 0)

    return enc.astype(np.float32, copy=False)


def sample_negatives(fa, peaks: pd.DataFrame, n: int,
                     window: int, seed: int) -> tuple[np.ndarray, list[str]]:
    pass


def build_dataset(cfg: DataConfig) -> dict:
    pass


def split_by_chromosome(X, y, chroms, test_chroms) -> tuple:
    pass
