import numpy as np
from pyliftover import LiftOver
from Bio.Seq import Seq
from scipy.sparse import csr_matrix
# lo = LiftOver('hg19', 'hg38')
# %%
# hg19 = Fasta('/home/xf2217/Projects/common/hg19.fasta')
# hg38 = Fasta('/home/xf2217/Projects/common/hg38.fa')
#%%
# class for sequence manipulation
class DNASequence(Seq):
    def __init__(self, seq, header = ''):
        self.header = header
        self.seq = seq.upper()
        self.one_hot_encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    
    def __repr__(self) -> str:
        return f"DNA sequence: {self.seq}"

    def get_reverse_complement(self):
        """
        Get the reverse complement of a DNA sequence
        """
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        return ''.join([complement[base] for base in self.seq[::-1]])
    
    def padding(self, left=0, right=0, target_length=0):
        """
        Pad a DNA sequence
        """
        if target_length == 0:
            return DNASequence('N' * left + self.seq + 'N' * right, self.header)
        elif target_length > len(self.seq):
            return DNASequence('N' * left + self.seq + 'N' * (target_length - len(self.seq)- left), self.header)
        elif target_length < len(self.seq):
            return DNASequence(self.seq[(len(self.seq) - target_length) // 2:(len(self.seq) + target_length) // 2], self.header)

    #attribute to get one-hot encoding
    @property
    def one_hot(self):
        """
        Get one-hot encoding of a DNA sequence
        """

        return csr_matrix(np.array([self.one_hot_encoding[base] for base in self.seq]))

