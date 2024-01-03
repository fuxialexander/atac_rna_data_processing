import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from pyliftover import LiftOver
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, vstack
import zarr
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
        self.seq = str(seq).upper().encode()
        self._data = str(seq).upper().encode() # convert DNA sequence to upper case and encode from ASCII to bytes

        self.one_hot_encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    

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
            return DNASequence('N' * left + self.seq.decode() + 'N' * right, self.header)
        elif target_length >= len(self.seq):
            return DNASequence('N' * left + self.seq.decode() + 'N' * (target_length - len(self.seq)- left), self.header)
        elif target_length < len(self.seq):
            return DNASequence(self.seq.decode()[(len(self.seq) - target_length) // 2:(len(self.seq) + target_length) // 2], self.header)
    
    def mutate(self, pos, alt):
        """
        Mutate a DNA sequence using biopython
        """
        from Bio.Seq import MutableSeq
        if type(pos) != int:
            pos = int(pos)
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
        return np.array([self.one_hot_encoding[base] for base in self.seq.decode()]).astype(np.int8).reshape(-1, 4)

    def save_zarr(self, zarr_file_path, 
              included_chromosomes=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7',
                                    'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13',
                                    'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                                    'chr20', 'chr21', 'chr22', 'chrX', 'chrY']
              ):
        """
        Save the genome sequence data in Zarr format.
        
        Args:
            zarr_file_path (str): Path to the Zarr file containing genome data.
            included_chromosomes (list): List of chromosomes to be included in the Zarr file.
        """
        zarr_file = zarr.open_group(zarr_file_path, 'w')
        for chr in tqdm(included_chromosomes):
            data = self.get_sequence(chr, 0, self.chrom_sizes[chr]).one_hot
            zarr_file.create_dataset(chr, data=data, chunks=(2000000, 4), 
                                    dtype='i4',
                                    compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))
        return

class DNASequenceCollection():
    """A collection of DNA sequences objects"""
    def __init__(self, sequences):
        self.sequences = sequences

    def __iter__(self):
        """for each sequence in the collection, convert to DNASequence object"""
        for seq in self.sequences:
            yield DNASequence(str(seq.seq), seq.id)


    
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

    def scan_motif(self, motifs, non_negative=True, raw = False):
        seqs = self.sequences
        # initialize the output list
        output = []
        # scan each sequence and add the results to the output list
        headers = []
        lengths = []
        sequences = []
        for s in seqs:
            sequences.append(str(s.seq))
            headers.append(s.header)
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
        
        if raw == True:
            return output
            
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
            sparse_matrix = csr_matrix(seq.one_hot)
            # add the sparse matrix to the list
            sparse_matrices.append(sparse_matrix)
        
        # concatenate the sparse matrices vertically
        sparse_matrix = vstack(sparse_matrices)
        
        # save the sparse matrix to a npz file with sequence length information in the filename
        save_npz(filename, sparse_matrix)

    def save_txt(self, filename):
        """Save the DNASequenceCollection object as a text file"""
        with open(filename, 'w') as f:
            for seq in self.sequences:
                f.write(seq.seq + '\n')

    def save_zarr(self, filename, chunks = (100, 2000, 4), target_length = 2000):
        """Save the one-hot encoding of a DNASequenceCollection object as a zarr file. Don't use sparse matrix, use compression"""
        # create a list to store the one-hot encoding
        one_hot = []
        
        # loop over the sequences and create a one-hot encoding for each sequence
        for seq in tqdm(self.sequences):
            # pad the sequence
            if len(seq.seq) != target_length:
                seq = seq.padding(left=0, target_length=target_length)
            # create the one-hot encoding
            one_hot.append(seq.one_hot)
        
        # concatenate the one-hot encoding vertically
        one_hot = np.stack(one_hot).astype(np.int8)
        
        # save the one-hot encoding to a zarr file
        zarr.save(filename, one_hot, chunks=chunks)
    

    def save_zarr_group(self, zarr_root, key, chunks=(100, 2000, 4), target_length=2000):
        """Save the one-hot encoding of a DNASequenceCollection object as a zarr group.

        Args:
            zarr_root (str): The root directory of the zarr storage.
            key (str): The key under which the data will be stored in the zarr group.
            chunks (tuple): The chunk size for zarr storage.
        """
        # create a list to store the one-hot encoding
        one_hot = []

        # loop over the sequences and create a one-hot encoding for each sequence
        for seq in tqdm(self.sequences):
            # pad the sequence
            if len(seq.seq) != target_length:
                seq = seq.padding(left=0, target_length=target_length)
            # create the one-hot encoding
            one_hot.append(seq.one_hot)
        
        # concatenate the one-hot encoding vertically
        one_hot = np.stack(one_hot).astype(np.int8)
        
        # Initialize a zarr group/store at the specified root
        zarr_group = zarr.open_group(zarr_root, mode='a')

        # save the one-hot encoding to the specified key in the zarr group
        zarr_group.create_dataset(key, data=one_hot, chunks=chunks, dtype='i1', compressor=zarr.Blosc(cname='zstd', clevel=3))
