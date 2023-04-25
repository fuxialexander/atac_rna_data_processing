import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from pyliftover import LiftOver
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, vstack

from atac_rna_data_processing.io.motif import (pfm_conversion, prepare_scanner,
                                               print_results)
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1


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
    
    def mutate(self, pos, alt):
        """
        Mutate a DNA sequence using biopython
        """
        from Bio.Seq import MutableSeq
        if len(alt) == 1:
            seq = MutableSeq(self.seq)
            seq[pos] = alt
        else: 
            seq = str(self.seq)
            seq = seq[0:pos] + alt + seq[pos+1:]
        return DNASequence(str(seq), self.header)

    #attribute to get one-hot encoding
    @property
    def one_hot(self):
        """
        Get one-hot encoding of a DNA sequence
        """

        return csr_matrix(np.array([self.one_hot_encoding[base] for base in self.seq]))

class DNASequenceCollection():
    """A collection of DNA sequences objects"""
    def __init__(self, sequences):
        self.sequences = sequences
    
    def from_fasta(filename):
        """
        Read a fasta file and create a DNASequenceCollection object using SeqIO.parse
        """
        return DNASequenceCollection(list(SeqIO.parse(filename, "fasta")))

    def mutate(self, pos_list, alt_list):
        """
        Mutate a DNASequenceCollection object
        """
        return DNASequenceCollection([seq.mutate(pos, alt) for seq, pos, alt in zip(self.sequences, pos_list, alt_list)])

    def scan_motif(self, motifs, non_negative=True):
        seqs = self.sequences
        # initialize the output list
        output = []
        # scan each sequence and add the results to the output list
        headers = []
        lengths = []
        sequences = []
        for s in seqs:
            sequences.append(str(s.seq))
            headers.append(s.id)
            lengths.append(len(str(s.seq)))

        # concatenate the sequences with 100 Ns between each sequence
        seq_cat = ("N" * 100).join(sequences)
        # get the list of sequence start and end positions in the concatenated sequence
        starts = np.cumsum([0] + lengths[:-1]) + 100 * np.arange(len(seqs))
        ends = starts + np.array(lengths)
        headers = np.array(headers)
        # scan the concatenated sequence
        results = motifs.scanner.scan(seq_cat)
        output = print_results(
            "", seq_cat, motifs.matrices, motifs.matrix_names, results
        )
        # convert the output list to a dataframe
        output = pd.DataFrame(
            output, columns=["header", "motif", "pos", "strand", "score", "seq"]
        )
        output["cluster"] = output.motif.map(motifs.motif_to_cluster)

        # assign header names in output dataframe based on 'pos' and starts/ends
        for i, h in enumerate(headers):
            output.loc[(output.pos >= starts[i]) & (output.pos < ends[i]), "header"] = h

        # remove the rows with multiple Ns
        output = output[~output.seq.str.contains("NN")]

        output = (
            output.groupby(["header", "pos", "cluster"])
            .score.max()
            .reset_index()
            .groupby(["header", "cluster"])
            .score.sum()
            .reset_index()
        )

        if non_negative:
            output.loc[output.score < 0, "score"] = 0

        motif_c = pd.CategoricalDtype(categories=motifs.cluster_names, ordered=True)
        seq_c = pd.CategoricalDtype(categories=headers, ordered=True)

        row = output.header.astype(seq_c).cat.codes
        col = output.cluster.astype(motif_c).cat.codes

        sparse_matrix = csr_matrix(
            (output["score"], (row, col)),
            shape=(seq_c.categories.size, motif_c.categories.size),
        )

        output = pd.DataFrame.sparse.from_spmatrix(
            sparse_matrix, index=seq_c.categories, columns=motif_c.categories
        )

        return output


    def save_npz(self, filename):
        """
        Save a DNASequenceCollection object as one-hot encoding in a sparse matrix in npz format,
        with sequence length information included in the filename
        """
        # create a list to store the sparse matrices
        sparse_matrices = []
        
        # loop over the sequences and create a sparse matrix for each one-hot encoding
        for seq in tqdm(self.sequences):
            # create the sparse matrix for the one-hot encoding
            sparse_matrix = seq.one_hot.tocsr()
            # add the sparse matrix to the list
            sparse_matrices.append(sparse_matrix)
        
        # concatenate the sparse matrices vertically
        sparse_matrix = vstack(sparse_matrices)
        
        # get the sequence length
        seq_length = len(self.sequences[0].seq)
        
        # append sequence length to the filename
        filename = f"{filename}_{seq_length}"
        
        # save the sparse matrix to a npz file with sequence length information in the filename
        save_npz(filename, sparse_matrix)