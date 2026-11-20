
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pyfaidx import Fasta


CANONICAL_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]

NARROWPEAK_COLS = [
    "chrom", "start", "end", "name", "score", "strand",
    "signalValue", "pValue", "qValue", "summit",
]


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
    pass


def extract_window(fa, chrom: str, center: int, width: int) -> np.ndarray:
    pass


def sample_negatives(fa, peaks: pd.DataFrame, n: int,
                     window: int, seed: int) -> tuple[np.ndarray, list[str]]:
    pass


def build_dataset(cfg: DataConfig) -> dict:
    pass


def split_by_chromosome(X, y, chroms, test_chroms) -> tuple:
    pass
