import numpy as np
import pandas as pd
import pyfaidx
import pathlib import Path
import dataclasses


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


def load_peaks(path: Path) -> pd.Dataframe:
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Peak file not found: {path}")

    df = pf.read_csv(
            path,
            sep="\t",
            header=None,
            names=NARROWPEAK_COLS,
            usecols=["chrom", "start", "end", "signalvalue", "summit"],
            dtype={"chrom": str})

    df



