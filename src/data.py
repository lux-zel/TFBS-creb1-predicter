
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
    """All knobs that affect dataset content. Used for cache-keying."""
    bed_path: Path = Path("data/raw/creb1_peaks.bed")
    fasta_path: Path = Path("data/raw/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    out_dir: Path = Path("data/processed")

    window: int = 500
    neg_per_pos: int = 1
    seed: int = 0

    test_chroms: tuple[str, ...] = ("1", "2")

    def cache_path(self) -> Path:
        return self.out_dir / (
            f"dataset_w{self.window}_n{self.neg_per_pos}_s{self.seed}.npz"
        )


def load_peaks(path: Path) -> pd.DataFrame:
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Peak file not found: {path}")

    df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=NARROWPEAK_COLS,
            usecols=["chrom", "start", "end", "signalValue", "summit"],
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


def _peak_center(row) -> int:
    if row.summit >= 0:
        return int(row.start + row.summit)
    return int(row.start + (row.end - row.start) // 2)


def sample_negatives(fa: Fasta, peaks: pd.DataFrame, n: int,
                     width: int, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)

    peak_by_chrom: dict[str, np.ndarray] = {}
    for c, sub in peaks.groupby("chrom"):
        arr = sub[["start", "end"]].to_numpy(dtype=np.int64)
        peak_by_chrom[c] = arr[np.argsort(arr[:, 0])]

    present = [c for c in CANONICAL_CHROMS if c in fa.keys()]
    lengths = np.array([len(fa[c]) for c in present], dtype=np.int64)

    if (lengths < width).all():
        raise ValueError("No chromosome is long enough to fit a window.")

    usable = lengths >= width
    present = [c for c, ok in zip(present, usable) if ok]
    weights = lengths[usable] / lengths[usable].sum()

    X = np.empty((n, 4, width), dtype=np.float32)
    chroms: list[str] = []

    max_attempts = n * 100
    attempts = 0
    filled = 0
    while filled < n:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Negative sampling failed after {max_attempts} attempts "
                f"({filled}/{n} filled). Chromosome may be too dense with peaks."
            )
        attempts += 1

        chrom = present[rng.choice(len(present), p=weights)]
        chrom_len = len(fa[chrom])
        start = int(rng.integers(0, chrom_len - width + 1))
        end = start + width

        pk = peak_by_chrom.get(chrom)
        if pk is not None and pk.size:
            starts = pk[:, 0]
            lo = np.searchsorted(pk[:, 1], start, side="right")
            hi = np.searchsorted(starts, end, side="left")
            if hi > lo:
                continue  

        seq = str(fa[chrom][start + 1: end + 1]).upper()
        X[filled] = one_hot(seq)
        chroms.append(chrom)
        filled += 1

    return X, chroms


def build_dataset(cfg: DataConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cache = cfg.cache_path()

    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {
            "X_pos": z["X_pos"],
            "chroms_pos": z["chroms_pos"],
            "X_neg": z["X_neg"],
            "chroms_neg": z["chroms_neg"],
        }

    if not Path(cfg.fasta_path).exists():
        raise FileNotFoundError(f"FASTA not found: {cfg.fasta_path}")

    peaks = load_peaks(cfg.bed_path)
    fa = Fasta(str(cfg.fasta_path), as_raw=True)

    centers = np.array([_peak_center(r) for r in peaks.itertuples()])
    X_pos = np.empty((len(peaks), 4, cfg.window), dtype=np.float32)
    for i, (c, center) in enumerate(zip(peaks["chrom"], centers)):
        X_pos[i] = extract_window(fa, c, int(center), cfg.window)
    chroms_pos = peaks["chrom"].to_numpy()

    n_neg = int(len(peaks) * cfg.neg_per_pos)
    X_neg, chroms_neg = sample_negatives(
        fa, peaks, n=n_neg, width=cfg.window, seed=cfg.seed
    )

    np.savez_compressed(
        cache,
        X_pos=X_pos,
        chroms_pos=chroms_pos,
        X_neg=X_neg,
        chroms_neg=np.array(chroms_neg, dtype=object),
    )

    return {
        "X_pos": X_pos,
        "chroms_pos": chroms_pos,
        "X_neg": X_neg,
        "chroms_neg": np.array(chroms_neg, dtype=object),
    }


def split_by_chromosome(data: dict, test_chroms: Iterable[str] = ("1", "2")) -> dict:
    test_set = set(test_chroms)

    
    pos_mask = np.array([c in test_set for c in data["chroms_pos"]])
    X_pos_tr = data["X_pos"][~pos_mask]
    X_pos_te = data["X_pos"][pos_mask]
    chroms_pos_tr = data["chroms_pos"][~pos_mask]
    chroms_pos_te = data["chroms_pos"][pos_mask]

    # Negatives
    neg_mask = np.array([c in test_set for c in data["chroms_neg"]])
    X_neg_tr = data["X_neg"][~neg_mask]
    X_neg_te = data["X_neg"][neg_mask]
    chroms_neg_tr = data["chroms_neg"][~neg_mask]
    chroms_neg_te = data["chroms_neg"][neg_mask]

    X_train = np.concatenate([X_pos_tr, X_neg_tr], axis=0)
    y_train = np.concatenate([
        np.ones(len(X_pos_tr), dtype=np.float32),
        np.zeros(len(X_neg_tr), dtype=np.float32),
    ])
    chroms_train = np.concatenate([chroms_pos_tr, chroms_neg_tr])

    X_test = np.concatenate([X_pos_te, X_neg_te], axis=0)
    y_test = np.concatenate([
        np.ones(len(X_pos_te), dtype=np.float32),
        np.zeros(len(X_neg_te), dtype=np.float32),
    ])
    chroms_test = np.concatenate([chroms_pos_te, chroms_neg_te])

    return {
        "X_train": X_train, "y_train": y_train, "chroms_train": chroms_train,
        "X_test": X_test, "y_test": y_test, "chroms_test": chroms_test,
    }
