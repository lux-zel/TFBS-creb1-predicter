''' import modules '''
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


