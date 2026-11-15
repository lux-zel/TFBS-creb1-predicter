# data_prep.py
import random
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from config import DATA_DIR, PEAK_FILE, GENOME_FILE, HALF_LEN, SEQ_LEN, N_NEGATIVES
import os

#check of de directory wel bestaat
os.makedirs(DATA_DIR, exist_ok=True)

#genoom laden
print("Loading genome...")
genome = Fasta(GENOME_FILE)
#krijg alle chromosoom namen zoals ze in het fasta bestand staan
genome_chroms = set(genome.keys())
print(f"Genome contains {len(genome_chroms)} chromosomes.")
#een paar voorbeelden
print("Example chromosome names:", list(genome_chroms)[:10])

#een dict met de lengtes van de chromosomen
chrom_lengths = {chrom: len(genome[chrom]) for chrom in genome_chroms}

#de peaks uit creb1 peaks laden
print("Loading peaks...")
peaks = pd.read_csv(PEAK_FILE, sep='\t', header=None,
                    names=['chrom','start','end','name','score','strand',
                           'signalValue','pValue','qValue','peak'])
peaks = peaks[peaks['peak'].notna()].reset_index(drop=True)
print(f"Total peaks with summit: {len(peaks)}")

# one hot encoding om de ACTG naar nummers te veranderen
def one_hot_encode(seq):
    """Convert DNA string to 4xL numpy array (float32)."""
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    one_hot = np.zeros((4, len(seq)), dtype=np.float32)
    for i, nt in enumerate(seq.upper()):
        if nt in mapping:
            one_hot[mapping[nt], i] = 1.0
    return one_hot

def extract_sequence(chrom, center):
    """Return one-hot encoded sequence of length SEQ_LEN centered at `center`.
       Returns None if the window falls outside the chromosome."""
    start = max(0, center - HALF_LEN)
    end = center + HALF_LEN

    if end > chrom_lengths[chrom]:
        return None
    seq = genome[chrom][start:end].seq
    if len(seq) == SEQ_LEN:
        return one_hot_encode(seq)
    return None

# extractie van positieve sequenties
print("Extracting positive sequences...")
positive_seqs = []
positive_coords = []    

for idx, row in peaks.iterrows():
    chrom = row['chrom']
  
    if chrom.startswith('chr'):
        chrom = chrom[3:]
 
    if chrom not in genome_chroms:
        continue

    center = row['start'] + int(row['peak'])
    seq_array = extract_sequence(chrom, center)
    if seq_array is not None:
        positive_seqs.append(seq_array)
        positive_coords.append((chrom, center))

positive_seqs = np.array(positive_seqs)
print(f"Extracted {len(positive_seqs)} positive sequences (after filtering).")

#maak een set van positieve intervals
pad = HALF_LEN
peak_intervals = set()
for chrom, center in positive_coords:
    peak_intervals.add((chrom, center - pad, center + pad))

def overlaps_peak(chrom, center):
    """Return True if the window centered at `center` overlaps any positive interval."""
    for pchrom, pstart, pend in peak_intervals:
        if pchrom == chrom and not (center + pad < pstart or center - pad > pend):
            return True
    return False

#genereer negatieve sequenties
print("Generating negative sequences...")
negative_seqs = []
attempts = 0
max_attempts = N_NEGATIVES * 10

while len(negative_seqs) < N_NEGATIVES and attempts < max_attempts:
    attempts += 1
 
    chrom = random.choice(list(genome_chroms))
    chrom_len = chrom_lengths[chrom]
  
    if chrom_len < SEQ_LEN:
        continue

    center = random.randint(HALF_LEN, chrom_len - HALF_LEN)

    if overlaps_peak(chrom, center):
        continue
    seq_array = extract_sequence(chrom, center)
    if seq_array is not None:
        negative_seqs.append(seq_array)

negative_seqs = np.array(negative_seqs)
print(f"Generated {len(negative_seqs)} negative sequences.")

# sla de data op
np.save(os.path.join(DATA_DIR, "positive_seqs.npy"), positive_seqs)
np.save(os.path.join(DATA_DIR, "negative_seqs.npy"), negative_seqs)
print("Data saved to data/positive_seqs.npy and data/negative_seqs.npy")