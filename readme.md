# TFBS-CREB1-Predicter

A 1D convolutional neural network that predicts whether a 500 bp DNA sequence contains a CREB1 binding site, trained on ENCODE ChIP-seq data and evaluated on held-out chromosomes.

## Overview

CREB1 (cAMP response element-binding protein 1) is a transcription factor that binds the cAMP response element (CRE) with the canonical motif `TGACGTCA`. This project trains a small 1D CNN to classify genomic windows as CREB1-bound (positive) or unbound (negative) from sequence alone.

**Headline result:** AUROC **0.987** on held-out chromosomes 1 and 2, with verified enrichment for the canonical CREB1 motif in the model's positives.

## Results

Evaluated on chromosomes 1 and 2, which were never seen during training (15,725 held-out sequences, 54% positive):

| Metric | Value |
|---|---|
| AUROC | **0.987** |
| AUPRC | 0.989 |
| Accuracy (θ = 0.5) | 0.948 |
| Precision | 0.959 |
| Recall | 0.944 |
| False positive rate | 4.7% |
| False negative rate | 5.6% |

Per-chromosome AUROC is consistent (chr1: 0.989, chr2: 0.986), ruling out chromosome-specific artifacts.

**Motif enrichment.** The model's positives contain the canonical CREB1 consensus `TGACGTCA` (33 hits in ~1,080 positives, **zero** hits in ~920 negatives) and its 1-bp variant `TGACGTAA` (22 hits, zero in negatives). This confirms the model learned the real binding motif, not background GC composition, a known failure mode for DNA-binding CNNs.

## Reproducing

```bash
git clone https://github.com/lux-zel/TFBS-creb1-predicter.git TFBS-creb1-predicter
cd TFBS-creb1-predicter
bash scripts/download_data.sh    # downloads ~3 GB genome + peak file
pip install -e .
jupyter lab
```

## Then open the notebooks in order:

1. `notebooks/00_data_sanity.ipynb` — verify the pipeline (~1 min with cache)
2. `notebooks/01_train.ipynb` — train the model (~15 min on GPU)
3. `notebooks/02_eval.ipynb` — evaluate on held-out chromosomes (~1 min)

Training without a GPU works but is slow (~2.5–3 hours on a 2-core CPU). An NVIDIA GPU (tested on Jetson Orin Nano Super) trains the full 20 epochs in ~15–20 minutes.

## Method

### Data

- **Peaks:** ENCODE CREB1 ChIP-seq in HepG2 cells (accession ENCSR112ALD), IDR-thresholded narrowPeak calls aligned to GRCh38. Downloaded via the GEO mirror `GSE127408`.
- **Reference genome:** Ensembl GRCh38 primary assembly (release 113).
- **Positives:** 45,725 peak-centered 500 bp windows, one-hot encoded as `(4, 500)` arrays.
- **Negatives:** 45,725 length-weighted random 500 bp windows from canonical chromosomes, rejection-sampled to avoid all peaks. Final dataset: 91,450 sequences, balanced 1:1.

### Model

A small 1D CNN (~1.4 M parameters):

```
Conv1d(4→64, k=11) → ReLU → Conv1d(64→64, k=11) → ReLU → MaxPool(4)
Conv1d(64→128, k=11) → ReLU → Conv1d(128→128, k=11) → ReLU → MaxPool(4)
Conv1d(128→256, k=11) → ReLU → Conv1d(256→256, k=11) → ReLU
AdaptiveAvgPool1d(1) → Dropout(0.3) → Linear(256→1)
```

The `AdaptiveAvgPool1d` makes the model input-length agnostic — the head works on any window size.

### Training

- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam, lr = 1e-3
- **Batch size:** 256
- **Epochs:** 20, with best-val checkpointing
- **Split:** train on chromosomes 3–22, X, Y; evaluate on chromosomes 1 and 2

Best validation loss was reached at **epoch 7** (val_loss = 0.125). Val loss rose after epoch 8 while train loss continued falling, classic overfitting. The checkpoint saved and evaluated is from epoch 7, not the final epoch.

### Important caveat on the headline AUROC

AUROC 0.987 is at the upper end of what's typically reported for TF binding CNNs (state-of-the-art models like Basenji and Enformer achieve 0.90–0.95). The most likely contributing factor is the **negative sampling strategy**: negatives are random genomic windows, which are easier to classify than negatives matched to positives on GC content or dinucleotide composition.

This doesn't mean the result is invalid, the motif enrichment check confirms real binding signal, but a stricter benchmark (GC-matched or dinucleotide-shuffled negatives) would produce a more conservative AUROC in the 0.92–0.95 range. The current setup is a reasonable "does this pipeline work" baseline, not a state-of-the-art claim.

## Project structure

```
TFBS-creb1-predicter/
├── data/
│   ├── raw/                          # BED + FASTA (gitignored)
│   └── processed/                    # cached .npz arrays + metrics.json
├── notebooks/
│   ├── 00_data_sanity.ipynb          # verify the pipeline
│   ├── 01_train.ipynb                # train the CNN
│   └── 02_eval.ipynb                 # evaluate on held-out chromosomes
├── scripts/
│   └── download_data.sh              # fetch BED + FASTA from GEO/Ensembl
├── src/
│   ├── data.py                       # BED/FASTA parsing, one-hot, negatives, splits
│   ├── model.py                      # Creb1CNN architecture
│   ├── train.py                      # training loop + CLI
│   ├── evaluate.py                   # metrics + CLI
│   └── notebook_setup.py             # CWD helper for notebooks
├── tests/
│   ├── test_data.py                  # pyfaidx offset, one-hot, filtering tests
│   └── test_model.py                 # shape and length-agnosticism tests
├── environment.yml                   # conda env spec
├── pyproject.toml                    # pip install -e . config
└── Makefile                          # `make download`, `make data`, `make test`
```

## Running from the terminal

Everything the notebooks do can also be run from the CLI:

```bash
# Build the cached dataset (one-time, ~5 min)
make data

# Train
python -m src.train --epochs 20 --batch-size 256 --lr 1e-3

# Evaluate
python -m src.evaluate --ckpt checkpoints/cnn.pt --out data/processed/metrics.json
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The tests focus on the two failure modes most likely to corrupt the pipeline silently:

- `test_data.py` — pyfaidx's 1-based indexing vs. BED's 0-based coordinates (an off-by-one here corrupts every positive sample), one-hot encoding correctness, negative sampling never overlapping peaks.
- `test_model.py` — output shape is `(batch,)` not `(batch, 1)` (the most common silent bug in PyTorch sequence models), and the model is length-agnostic.


## Dependencies

Core: `torch`, `numpy`, `pandas`, `pyfaidx`, `scikit-learn`, `matplotlib`, `jupyterlab`.

Tested with Python 3.11/3.12, PyTorch 2.5+ (CPU) and 2.15+cu132 (Jetson Orin Nano, JetPack 7).


## Citation

If you use this code, please cite the underlying data sources:

- **ENCODE Project Consortium** (2012). An integrated encyclopedia of DNA elements in the human genome. *Nature*, 489, 57–74.
- **Consortium, T. E. P.** (2020). Expanded encyclopaedias of DNA elements in the human and mouse genomes. *Nature*, 583, 699–710.



