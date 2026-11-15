import os
import torch

DATA_DIR = 'data'
PEAK_FILE = os.path.join(DATA_DIR, 'creb1_peaks.bed')
GENOME_FILE = os.path.join(DATA_DIR, 'Homo_sapiens.GRCh38.dna.primary_assembly.fa')


HALF_LEN = 100
SEQ_LEN = 2 * HALF_LEN

N_NEGATIVES = 10000

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'