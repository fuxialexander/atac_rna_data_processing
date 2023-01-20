import numpy as np
import pandas as pd
import seaborn as sns
from pyfaidx import Fasta
from tqdm import tqdm
from pyliftover import LiftOver
from pyranges import PyRanges
from atac_rna_data_processing.sequence import DNASequence
from atac_rna_data_processing.motif import pfm_conversion, prepare_scanner, print_results
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype


class Genome(object):
    def __init__(self, assembly: str, fasta_file: str) -> None:
        self.fasta_file = fasta_file
        self.assembly = assembly
        self.genome_seq = Fasta(fasta_file)
        if list(self.genome_seq.keys())[0].startswith('chr'):
            self.chr_suffix = 'chr'
        else:
            self.chr_suffix = ''

    def __repr__(self) -> str:
        return f"Genome: {self.assembly} with fasta file: {self.fasta_file}"

    def normalize_chromosome(self, chromosome):
        """
        Normalize chromosome name
        """
        if str(chromosome).startswith('chr'):
            chromosome = str(chromosome)[3:]

        return self.chr_suffix + str(chromosome)

    def get_sequence(self, chromosome, start, end, strand='+'):
        """
        Get the sequence of the genomic region
        """
        chromosome = self.normalize_chromosome(chromosome)
        if strand == '-':
            return DNASequence(self.genome_seq[chromosome][start:end].seq.complement(), header=f"{chromosome}_{start}_{end}")
        else:
            return DNASequence(self.genome_seq[chromosome][start:end].seq, header=f"{chromosome}_{start}_{end}")


class GenomicRegion(object):
    def __init__(self, genome: Genome, chromosome: str, start: int, end: int, strand: str = '+'):
        self.genome = genome
        self.chromosome = chromosome
        self.start = start
        self.end = end
        self.strand = strand

    def __repr__(self) -> str:
        return f"[{self.genome.assembly}]{self.chromosome}:{self.start}-{self.end}"

    @property
    def sequence(self):
        """
        Get the sequence of the genomic region
        """
        return self.genome.get_sequence(self.chromosome, self.start, self.end, self.strand)

    @property
    def motif_score(self, motif):
        """
        Get the motif score of the genomic region
        """
        return motif.score(self.sequence)

    def lift_over(self, target_genome, lo):
        """
        Lift over the genomic region to another genome assembly using lo object
        User need to provide the correct lo object
        """
        chromosome, start, end = lo.convert_coordinate(
            self.chromosome, self.start, self.end)
        if chromosome:
            return GenomicRegion(target_genome, chromosome, start, end, self.strand)
        else:
            return None

    def get_flanking_region(self, upstream, downstream):
        """
        Get the flanking region of the genomic region
        """
        return GenomicRegion(self.genome, self.chromosome, self.start - upstream, self.end + downstream, self.strand)


class GenomicRegionCollection(PyRanges):
    """List of GenomicRegion objects"""

    def __init__(self, genome,
                 df=None,
                 chromosomes=None,
                 starts=None,
                 ends=None,
                 strands=None,
                 int64=False,
                 copy_df=True):
        super().__init__(df,
                        chromosomes,
                        starts,
                        ends,
                        strands,
                        int64,
                        copy_df)
        self.genome = genome

    def __repr__(self) -> str:
        return f"GenomicRegionCollection with {len(self.df)} regions"

    def to_bed(self, file_name):
        """
        Save the genomic region collection to a bed file
        """
        self.df[['Chromosome', 'Start', 'End', 'Strand']].to_csv(
            file_name, sep='\t', header=False, index=False)

    def center_expand(self, target_size):
        """
        Expand the genomic region collection from peak center
        """
        peak_center = (self.df['End'] + self.df['Start']) // 2
        Start = peak_center - target_size//2
        End = peak_center + target_size//2
        if 'Strand' not in self.df.columns:
            return GenomicRegionCollection(genome=self.genome, df=pd.DataFrame({'Chromosome': self.df['Chromosome'], 'Start': Start, 'End': End}))
        else:
            return GenomicRegionCollection(genome=self.genome, df=pd.DataFrame({'Chromosome': self.df['Chromosome'], 'Start': Start, 'End': End, 'Strand': self.df['Strand']}))

    # generator of GenomicRegion objects
    def __iter__(self):
        for _, row in self.df.iterrows():
            if 'Strand' not in row:
                yield GenomicRegion(row['genome'], row['Chromosome'], row['Start'], row['End'], '+')
            else:
                yield GenomicRegion(row['genome'], row['Chromosome'], row['Start'], row['End'], row['Strand'])

    def __getitem__(self, val):
        if isinstance(val, int):
            row = self.df.iloc[val]
            if 'Strand' not in row:
                return GenomicRegion(row['genome'], row['Chromosome'], row['Start'], row['End'], '+')
            else:
                return GenomicRegion(row['genome'], row['Chromosome'], row['Start'], row['End'], row['Strand'])
        else:
            return GenomicRegionCollection(self.genome.iloc[val], self.df.iloc[val])

    def collect_sequence(self, mutations=None, upstream=0, downstream=0):
        """
        Collect the sequence of the genomic regions
        """
        if mutations is None:
            return [region.get_flanking_region(upstream, downstream).sequence for region in iter(self)]
        else:
            pass

    def scan_motif(self, motifs, mutations = None, non_negative=True, upstream=0, downstream=0):
        """
        Scan motif in sequence using MOODS.

        Parameters
        ----------
        seqs: List[Tuple[str, str]]
            A list of tuples containing the header and sequence for each input sequence.
        scanner: MOODS.Scanner
            The MOODS scanner to use for scanning the sequences.
        diff: bool, optional
            Whether to calculate the difference between the scores for the alternate and reference sequences. Defaults to False.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the results of the scan. If `diff` is True, the dataframe will include columns for the difference in scores and the cluster of the motif.
        """
        seqs = self.collect_sequence(mutations, upstream, downstream)
        # initialize the output list
        output = []

        # scan each sequence and add the results to the output list
        headers = []
        lengths = []
        sequences = []
        for s in seqs:
            sequences.append(s.seq)
            headers.append(s.header)
            lengths.append(len(s.seq))

        # concatenate the sequences with 100 Ns between each sequence
        seq_cat = ('N'*100).join(sequences)
        # get the list of sequence start and end positions in the concatenated sequence
        starts = np.cumsum([0] + lengths[:-1]) + 100*np.arange(len(seqs))
        ends = starts + np.array(lengths)
        headers = np.array(headers)
        # scan the concatenated sequence
        results = motifs.scanner.scan(seq_cat)
        output = print_results("", seq_cat, motifs.matrices,
                                motifs.matrix_names, results)
        # convert the output list to a dataframe
        output = pd.DataFrame(
            output, columns=['header', 'motif', 'pos', 'strand', 'score', 'seq'])
        output['cluster'] = output.motif.map(motifs.motif_to_cluster)

        # assign header names in output dataframe based on 'pos' and starts/ends
        for i, h in enumerate(headers):
            output.loc[(output.pos >= starts[i]) & (output.pos < ends[i]), 'header'] = h

        # remove the rows with multiple Ns
        output = output[~output.seq.str.contains('NN')]

        output = (
            output
            .groupby(['header', 'pos',  'cluster']).score.max()
            .reset_index()
            .groupby(['header', 'cluster']).score.sum()
            .reset_index()
        )

        if non_negative:
            output.loc[output.score<0, 'score'] = 0

        motif_c = CategoricalDtype(categories=motifs.cluster_names, ordered=True)
        seq_c = CategoricalDtype(categories=headers, ordered=True)


        row = output.header.astype(seq_c).cat.codes
        col = output.cluster.astype(motif_c).cat.codes

        sparse_matrix = csr_matrix(
            (output['score'], (row, col)), 
            shape=(seq_c.categories.size, motif_c.categories.size)
            )


        output = pd.DataFrame.sparse.from_spmatrix(
            sparse_matrix, 
            index=seq_c.categories, 
            columns=motif_c.categories)

        return output