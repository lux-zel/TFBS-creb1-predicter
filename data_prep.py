''' import modules '''
import random
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from config import *
import os

''' check of de directory wel bestaat'''
os.makedirs(DATA_DIR, exist_ok=True)

''' genome laden '''
genome = Fasta(GENOME_FILE)

''' de peaks laden '''
peaks = pd.read_csv(PEAK_FILE, sep='\t', header=None,
                    names=['chrom','start','end','name','score','strand',
                           'signalValue','pValue','qValue','peak'])
peaks = peaks[peaks['peak'].notna()].reset_index(drop=True)


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


