import numpy as np
from pyliftover import LiftOver
from Bio.Seq import Seq

# lo = LiftOver('hg19', 'hg38')
# %%
# hg19 = Fasta('/home/xf2217/Projects/common/hg19.fasta')
# hg38 = Fasta('/home/xf2217/Projects/common/hg38.fa')
#%%
# class for sequence manipulation
class DNASequence(Seq):
    def __init__(self, seq, header = ''):
        self.header = header
        self.seq = seq
        self.one_hot_encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    
    def __repr__(self) -> str:
        return f"DNA sequence: {self.seq}"

    def get_reverse_complement(self):
        """
        Get the reverse complement of a DNA sequence
        """
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        return ''.join([complement[base] for base in self.seq[::-1]])
    
    #attribute to get one-hot encoding
    @property
    def one_hot(self):
        """
        Get one-hot encoding of a DNA sequence
        """

        return np.array([self.one_hot_encoding[base] for base in self.seq])

