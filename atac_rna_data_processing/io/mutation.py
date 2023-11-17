# This specify the code to
# 1. read variants from VCF files
# 2. read Structure variants from bedpe files
# 3. Manipulate the given sequence or bed files to accomodate the variants
import random
import re
import time
import pandas as pd
import requests
from pysam import VariantFile
from tqdm import tqdm

import os
import tabix

from atac_rna_data_processing.io.region import GenomicRegionCollection
from atac_rna_data_processing.io.sequence import (DNASequence,
                                                  DNASequenceCollection)
import concurrent.futures
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np

from subprocess import Popen, PIPE
import subprocess

def bgzip(filename):
    """Call bgzip to compress a file."""
    Popen(['bgzip', '-f', filename])

def tabix_index(filename,
        preset="gff", chrom=1, start=4, end=5, skip=0, comment="#"):
    """Call tabix to create an index for a bgzip-compressed file."""
    Popen(['tabix', '-p', preset, '-s', chrom, '-b', start, '-e', end,
        '-S', skip, '-c', comment])


def tabix_query(filename, chrom, start, end, output_file, with_header=True):
    """
    Calls tabix to query a VCF file and saves the output to a file.

    Args:
    filename (str): The path to the VCF file.
    chrom (str): The chromosome.
    start (int): The start position of the query.
    end (int): The end position of the query.
    output_file (str): The file to save the output.
    """
    query = f"{chrom}:{start}-{end}"
    
    # Construct the tabix command
    command = ["tabix", filename, query]

    if with_header:
        command.append("-h")

    # output_file = os.path.join(base_dir, output_file)

    try:
        with open(output_file, "w") as f:
            # Execute the command and redirect the output to the file
            subprocess.run(command, stdout=f, check=True)

        # compress the output file
        bgzip(output_file)
        # index the output file
        tabix_index(output_file + ".gz")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while querying: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return output_file + ".gz"


def read_gwas_catalog(genome, gwas_catalog_csv_file):
    """Read GWAS catalog
    Args:
        gwas_catalog_csv_file: GWAS catalog file in csv format.
    """
    gwas = pd.read_csv(gwas_catalog_csv_file, sep='\t')
    chrom = gwas['CHR_ID'].astype(str).apply(lambda x: 'chr'+x)
    risk_allele = gwas['STRONGEST SNP-RISK ALLELE'].apply(
        lambda x: x.split('-')[1])
    variants = pd.DataFrame(
        {'Chromosome': chrom,
         'Start': gwas['CHR_POS'] - 1,
         'End': gwas['CHR_POS'],
         'Alt': risk_allele,
         'RSID': gwas['SNPS'],
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


def fetch_rsid_data(server, rsid, max_retries=5):
    """Fetch RSID data with retry mechanism for rate limiting."""
    ext = f"/variation/human/{rsid}?"
    for i in range(max_retries):
        try:
            r = requests.get(server+ext, headers={"Content-Type": "application/json"})
            r.raise_for_status()
            decoded = pd.DataFrame(r.json()['mappings'])
            decoded['RSID'] = rsid
            return decoded
        except requests.exceptions.HTTPError as err:
            if r.status_code == 429 and i < max_retries - 1:  # Too Many Requests
                wait_time = (2 ** i) + random.random()
                time.sleep(wait_time)
            else:
                raise err

def read_rsid_parallel(genome, rsid_file, num_workers=10):
    """Read VCF file, only support hg38
    """
    rsid_list = np.loadtxt(rsid_file, dtype=str)
    server = "http://rest.ensembl.org"
    df = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_rsid = {executor.submit(fetch_rsid_data, server, rsid): rsid for rsid in tqdm(rsid_list)}
        for future in concurrent.futures.as_completed(future_to_rsid):
            df.append(future.result())

    df = pd.concat(df).query('~location.str.contains("CHR")').query('assembly_name=="GRCh38"')
    df['Start'] = df['start']-1
    df['End'] = df['start']
    df['Chromosome'] = df.seq_region_name.apply(lambda x: 'chr'+x)
    df['Ref'] = df.allele_string.apply(lambda x: x.split('/')[0])
    df['Alt'] = df.allele_string.apply(lambda x: x.split('/')[1])

    return Mutations(genome, df[['Chromosome', 'Start', 'End', 'Ref', 'Alt', 'RSID']])

def read_rsid(genome, rsid_file):
    """Read VCF file, only support hg38
    """
    import numpy as np
    rsid_list = np.loadtxt(rsid_file, dtype=str)
    server = "http://rest.ensembl.org"
    df = []
    for rsid in tqdm(rsid_list):
        ext = f"/variation/human/{rsid}?"
        r = requests.get(
            server+ext, headers={"Content-Type": "application/json"})
        if not r.ok:
            r.raise_for_status()
        decoded = pd.DataFrame(r.json()['mappings'])
        decoded['RSID'] = rsid
        df.append(decoded)
    df = pd.concat(df).query('~location.str.contains("CHR")').query(
        'assembly_name=="GRCh38"')
    df['Start'] = df['start']-1
    df['End'] = df['start']
    df['Chromosome'] = df.seq_region_name.apply(lambda x: 'chr'+x)
    df['Ref'] = df.allele_string.apply(lambda x: x.split('/')[0])
    df['Alt'] = df.allele_string.apply(lambda x: x.split('/')[1])

    return Mutations(genome, df[['Chromosome', 'Start', 'End', 'Ref', 'Alt', 'RSID']])


class Mutations(GenomicRegionCollection):
    """Class to handle mutations
    """

    def __init__(self, genome, df):
        super().__init__(genome, df)
        self.collect_ref_sequence(30,30)
        self.collect_alt_sequence(30,30)
        return

    def collect_ref_sequence(self, upstream=0, downstream=0):
        """Collect reference sequences centered at the mutation sites
        """
        self.Ref_seq = [s.seq for s in super().collect_sequence(
            upstream=upstream, downstream=downstream).sequences]

    def collect_alt_sequence(self, upstream=0, downstream=0):
        """Collect alternative sequences centered at the mutation sites
        """
        if self.Ref_seq is None:
            self.collect_ref_sequence(upstream, downstream)
        n_mut = len(self.Ref_seq)
        Alt_seq = DNASequenceCollection(
            [DNASequence(s) for s in self.Ref_seq.values])
        Alt_seq = Alt_seq.mutate([upstream] * n_mut, self.Alt.values)
        Alt_seq = [s.seq for s in Alt_seq.sequences]
        self.Alt_seq = Alt_seq

    def get_motif_diff(self, motif):
        """Get motif difference between reference and alternative sequences
        """
        Alt_seq = DNASequenceCollection(
            [DNASequence(row.Alt_seq, row.RSID+'_'+row.Alt) for i, row in self.df.iterrows()])
        Ref_seq = DNASequenceCollection(
            [DNASequence(row.Ref_seq, row.RSID+'_'+row.Ref) for i, row in self.df.iterrows()])
        return {'Alt': Alt_seq.scan_motif(motif),
                'Ref': Ref_seq.scan_motif(motif)}
    
class MutationsInCellType(object):
    """Class to handle mutations in a specific cell type
    """
    def __init__(self, genome, df, cell_type):
        from pyranges import PyRanges as pr
        self.celltype = cell_type
        df = pr(cell_type.peak_annot).join(pr(df)).df
        # keep only variants with one base change
        df = df.query('Ref.str.len()==1 & Alt.str.len()==1')
        df['upstream'] = df.Start_b - df.Start
        df['downstream'] = df.End - df.End_b
        self.mut = GenomicRegionCollection(genome, df)
        self.upstream = self.mut.df.upstream.values
        self.downstream = self.mut.df.downstream.values
        self.Alt = self.mut.df.Alt.values
        self.Ref = self.mut.df.Ref.values
        # self.get_original_input()
        # self.get_altered_input()

    def get_original_input(self, motif):
        self.Ref_peak_seq = [s.seq for s in self.mut.collect_sequence(
            upstream=0, downstream=0).sequences]
        Ref_peak_seq = DNASequenceCollection(
            [DNASequence(self.Ref_peak_seq[i], row.RSID+'_'+row.Ref) for i, row in self.mut.df.iterrows()])
        self.Ref_input = Ref_peak_seq.scan_motif(motif)

    def get_altered_input(self, motif):
        if self.Ref_peak_seq is None:
            self.get_original_input()
        n_mut = len(self.Ref_peak_seq)
        Alt_peak_seq = DNASequenceCollection(
            [DNASequence(s) for s in self.Ref_peak_seq])
        Alt_peak_seq = Alt_peak_seq.mutate(list(self.upstream), self.Alt)
        Alt_peak_seq = [s.seq for s in Alt_peak_seq.sequences]
        self.Alt_peak_seq = Alt_peak_seq
        Alt_peak_seq = DNASequenceCollection(
            [DNASequence(self.Alt_peak_seq[i], row.RSID+'_'+row.Alt) for i, row in self.mut.df.iterrows()])
        self.Alt_input = Alt_peak_seq.scan_motif(motif)

    def predict_expression(self, rsid, gene, center, N, inf_model=None):
        """
        Calculate expression predictions for original and altered cell states based on rsid and gene.

        Args:
        rsid (str): Reference SNP ID.
        gene (str): Gene name.
        self (object): An instance of the self class, containing data and methods for cell mutations.
        center (int): Center position for slicing the data matrix.
        N (int): The size of the slice from the data matrix.

        Returns:
        tuple: A tuple containing expression predictions for the original and altered states.
        """
        # Calculate new motif
        import torch
        if inf_model is None:
            import sys
            sys.path.append('/manitou/pmg/users/xf2217/get_model/')
            from inference import InferenceModel
            import torch
            checkpoint_path = '/manitou/pmg/projects/resources/get_interpret/pretrain_finetune_natac_fetal_adult.pth'
            inf_model = InferenceModel(checkpoint_path, 'cuda')

        ref = self.mut.df.query('RSID==@rsid').Ref.values[0]
        alt = self.mut.df.query('RSID==@rsid').Alt.values[0]
        new_motif = (self.Alt_input.loc[f'{rsid}_{alt}'].sparse.to_dense().values + 0.01) / \
                    (self.Ref_input.loc[f'{rsid}_{ref}'].sparse.to_dense().values + 0.01)

        # Determine start and end indices based on the gene TSS
        gene_tss_info = self.celltype.get_gene_tss(gene)[0]
        start = gene_tss_info.peak_id - center
        end = start + N

        # Get strand information
        strand_idx = gene_tss_info.strand

        # Process original matrix
        original_matrix = self.celltype.input_all[start:end].toarray()
        # atac = original_matrix[:, 282].copy()
        original_matrix[:, 282] = 1

        # Process altered matrix
        # idx_altered = self.mut.df.query('RSID==@rsid').values[0][0] - start
        # print(idx_altered)
        altered_matrix = original_matrix.copy()
        altered_matrix[:, 0:282] = new_motif * altered_matrix[:, 0:282]

        # Create tensors for prediction
        original = torch.Tensor(original_matrix).unsqueeze(0).to(inf_model.device)
        altered = torch.Tensor(altered_matrix).unsqueeze(0).to(inf_model.device)
        seq = torch.randn(1, N, 283, 4).to(inf_model.device)  # Dummy seq data
        tss_mask = torch.ones(1, N).to(inf_model.device)  # Dummy TSS mask
        ctcf_pos = torch.ones(1, N).to(inf_model.device)  # Dummy CTCF positions

        # Predict expression
        _, original_exp = inf_model.predict(original, seq, tss_mask, ctcf_pos)
        _, altered_exp = inf_model.predict(altered, seq, tss_mask, ctcf_pos)

        # Calculate and return the expression predictions
        original_pred = 10 ** (original_exp[0, center, strand_idx].item()) - 1
        altered_pred = 10 ** (altered_exp[0, center, strand_idx].item()) - 1

        return original_pred, altered_pred

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


# class GnomAD:
#     """
#     Class to handle downloading of gnomAD data.
#     """

#     def __init__(self, gnomad_base_url):
#         self.gnomad_base_url = gnomad_base_url
#     ``