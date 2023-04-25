### This specify the code to
# 1. read variants from VCF files
# 2. read Structure variants from bedpe files
# 3. Manipulate the given sequence or bed files to accomodate the variants

import pandas as pd
from pysam import VariantFile

from atac_rna_data_processing.io.region import GenomicRegionCollection
from atac_rna_data_processing.io.sequence import DNASequence, DNASequenceCollection

def read_gwas_catalog(genome, gwas_catalog_csv_file):
    """Read GWAS catalog
    Args:
        gwas_catalog_csv_file: GWAS catalog file in csv format.
    """
    gwas = pd.read_csv(gwas_catalog_csv_file, sep='\t')
    chrom = gwas['CHR_ID'].astype(str).apply(lambda x: 'chr'+x)
    risk_allele = gwas['STRONGEST SNP-RISK ALLELE'].apply(lambda x: x.split('-')[1])
    variants = pd.DataFrame(
        {'Chromosome': chrom, 
        'Start': gwas['CHR_POS'] -1, 
        'End': gwas['CHR_POS'], 
        'Alt': risk_allele, 
        'Gene': gwas['MAPPED_GENE'], 
        'Trait': gwas['MAPPED_TRAIT'],
        'P-value': gwas['P-VALUE'],
        })
    variants = variants.drop_duplicates()
    grc = GenomicRegionCollection(genome, variants)
    variants['Ref'] = [s.seq for s in grc.collect_sequence().sequences]
    # filter out variants with same risk allele and reference allele
    variants = variants.query('Ref != Alt').reset_index(drop=True)
    return Mutations(genome, variants)

def read_vcf(self):
    """Read VCF file
    """
    pd.read_csv(self.vcf_file, sep='\t', header=None)
    
    return
    
class Mutations(GenomicRegionCollection):
    """Class to handle mutations
    """
    def __init__(self, genome, df):
        super().__init__(genome, df)
        self.collect_ref_sequence(20)
        self.collect_alt_sequence(20)
        return 

    
    def collect_ref_sequence(self, expand=20):
        """Collect reference sequences centered at the mutation sites
        """
        self.Ref_seq = [s.seq for s in super().collect_sequence(upstream=expand, downstream=expand).sequences]

    def collect_alt_sequence(self, expand=20):
        """Collect alternative sequences centered at the mutation sites
        """
        if self.Ref_seq is None:
            self.collect_ref_sequence(expand)
        n_mut = len(self.Ref_seq)
        Alt_seq = DNASequenceCollection([DNASequence(s) for s in self.Ref_seq.values])
        print(Alt_seq)
        print(self.Alt.values)
        Alt_seq = Alt_seq.mutate([expand] * n_mut, self.Alt.values)
        Alt_seq = [s.seq for s in Alt_seq.sequences]
        self.Alt_seq = Alt_seq

class SVs(object):
    """Class to handle SVs
    """
    def __init__(self, bedpe_file, genome):
        self.genome = genome
        self.bedpe_file = bedpe_file
        self.bedpe = self.read_bedpe()
        return
    
    def read_bedpe(self):
        """Read bedpe file
        """
        pd.read_csv(self.bedpe_file, sep='\t', header=None)
        return