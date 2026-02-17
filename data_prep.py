#import modules data_prep.py.
import random
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from config import *
import os


#check of de directory wel bestaat.
os.makedirs(DATA_DIR, exist_ok=True)


#genome laden.
genome = Fasta(GENOME_FILE)


#de peaks laden.
peaks = pd.read_csv(PEAK_FILE, sep='\t', header=None,
                    names=['chrom','start','end','name','score','strand',
                           'signalValue','pValue','qValue','peak'])
peaks = peaks[peaks['peak'].notna()].reset_index(drop=True)


#het encoding algoritme om van seq een dataset te maken waarmee het nn model kan werken.
def one_hot_encode(seq):
    
    mapping = {'A':0,
               'C':1,
               'G':2,
               'T':3}
    
    one_hot = np.zeros((4, len(seq)), dtype=np.float32)

    for i, nt in enumerate(seq.upper()):
        if nt in mapping:
            one_hot[mapping[nt], i] = 1.0
    return one_hot


def extract_sequence(chrom, center):
    
    start = max(0, center - HALF_LEN)
    end = center + HALF_LEN
    seq = genome[chrom][start:end].seq
    
    if len(seq) == SEQ_LEN:
        return one_hot_encode(seq)
    return None


#positieve sequenties krijgen.
positive_seqs = []
positive_coords = []

for idx, row in peaks.iterrows():
    
    center = row['start'] + int(row['peak'])
    seq_array = extract_sequence(row['chrom'], center)
    if seq_array is not None:
        positive_seqs.append(seq_array)
        positive_coords.append((row['chrom'], center))

positive_seqs = np.array(positive_seqs)


#hardcoded chromosoom lengtes voor het checken van positie.
chrom_sizes = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559,
    'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979,
    'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
    'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
    'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285,
    'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983,
    'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415
}


#initialiseren van een set peak intervals voor overlappende data te checken.
peak_intervals = set()
pad = HALF_LEN

for chrom, center in positive_coords:
    peak_intervals.add((chrom, center - pad, center + pad))

def overlaps_peak(chrom, center):
    
    for pchrom, pstart, pend in peak_intervals:
        if pchrom == chrom and not  (center + pad < pstart or center - pad > pend):
            return True
        return False
    

#genereer negatieve sequenties.
negative_seqs = []
attempts = 0
max_attempts = N_NEGATIVES * 10

while len(negative_seqs) < N_NEGATIVES and attempts < max_attempts:
    
    attempts += 1
    chrom = random.choice(list(chrom_sizes.keys()))
    center = random.randint(HALF_LEN, chrom_sizes[chrom] - HALF_LEN)

    if overlaps_peak(chrom, center):
        continue
    
    seq_array = extract_sequence(chrom, center)

    if seq_array is not None:
        negative_seqs.append(seq_array)

negative_seqs = np.array(negative_seqs)


#schrijf de geprepereerde data naar file
np.save(os.path.join(DATA_DIR, 'positive_seqs.npy'), positive_seqs)
np.save(os.path.join(DATA_DIR, 'negative_seqs.npy'), negative_seqs)