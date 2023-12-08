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
import concurrent.futures
import requests
from glob import glob
from itertools import repeat
import pandas as pd
from pyranges import PyRanges as pr
from tqdm import tqdm
import numpy as np
from subprocess import Popen, PIPE
import subprocess
from multiprocessing import Pool, get_context

from atac_rna_data_processing.config.load_config import load_config
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
from atac_rna_data_processing.io.celltype import GETCellType
from atac_rna_data_processing.io.region import *
from atac_rna_data_processing.io.sequence import (DNASequence,
                                                  DNASequenceCollection)
from get_model.inference import InferenceModel


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

def predict_celltype_exp(
    cell_id, 
    get_config_path,
    celltype_annot_dict, 
    variants_rsid,
    genome,
    motif,
    inf_model,
):
    get_config = load_config(get_config_path)
    get_config.celltype.jacob=False
    get_config.celltype.num_cls=2
    get_config.celltype.input=True
    get_config.celltype.embed=False
    get_config.assets_dir=''
    get_config.s3_file_sys=''
    get_config.celltype.data_dir = '/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/'
    get_config.celltype.interpret_dir='/manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac'

    get_celltype = GETCellType(cell_id, get_config)
    cell_type = celltype_annot_dict[cell_id]
    if pr(get_celltype.peak_annot).join(pr(variants_rsid.df)).df.empty:
        return [cell_type, 1], get_celltype, cell_mut
        
    cell_mut = MutationsInCellType(genome, variants_rsid.df, get_celltype)
    cell_mut.get_original_input(motif)
    cell_mut.get_altered_input(motif)
    ref_exp, alt_exp = cell_mut.predict_expression('rs55705857', 'MYC', 100, 200, inf_model=inf_model)
    return [cell_type, alt_exp/ref_exp], get_celltype, cell_mut


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


class CellMutCollection(object):
    """Collection of MutationsInCellTypes objects"""

    def __init__(
            self,
            celltype_annot_path="/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data/cell_type_pretrain_human_bingren_shendure_apr2023.txt",
            model_ckpt_path="/manitou/pmg/projects/resources/get_interpret/pretrain_finetune_natac_fetal_adult.pth",
            variants_path="/manitou/pmg/users/xf2217/gnomad/myc.tad.vcf.gz",
            celltype_path="/manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac/",
            get_config_path="/manitou/pmg/users/xf2217/atac_rna_data_processing/atac_rna_data_processing/config/GET",
            working_dir="/manitou/pmg/users/xf2217/interpret_natac/",
            num_workers=10,
        ):
        celltype_annot = pd.read_csv(celltype_annot_path)
        self.celltype_annot_dict = celltype_annot.set_index('id').celltype.to_dict()
        self.cell_ids = [os.path.basename(cell_id) for cell_id in sorted(glob(f"{celltype_path}/*"))]

        self.ckpt_path = model_ckpt_path
        self.working_dir = working_dir
        self.num_workers = num_workers
        self.get_config_path = get_config_path

        self.inf_model = InferenceModel(self.ckpt_path, 'cuda')
        self.genome = Genome('hg38', self.working_dir + "/hg38.fa")
        self.motif = NrMotifV1.load_from_pickle(working_dir + "/NrMotifV1.pkl")
        self.variants_rsid = read_rsid_parallel(self.genome, working_dir + 'myc_rsid.txt', 5)
        self.normal_variants = self.load_normal_filter_normal_variants(variants_path)
        
        self.celltype_to_get_celltype = {}
        self.celltype_to_mut_celltype = {}
    
    def load_normal_filter_normal_variants(self, variants_path):
        normal_variants = pd.read_csv(variants_path, sep='\t', comment='#', header=None)
        normal_variants.columns = ['Chromosome', 'Start', 'RSID', 'Ref', 'Alt', 'Qual', 'Filter', 'Info']
        normal_variants['End'] = normal_variants.Start
        normal_variants['Start'] = normal_variants.Start-1
        normal_variants = normal_variants[['Chromosome', 'Start', 'End', 'RSID', 'Ref', 'Alt', 'Qual', 'Filter', 'Info']]
        normal_variants = normal_variants.query('Ref.str.len()==1 & Alt.str.len()==1')
        normal_variants['AF'] = normal_variants.Info.transform(lambda x: float(re.findall(r'AF=([0-9e\-\.]+)', x)[0]))
        return normal_variants

    def predict_all_celltype_expression(self):
        with get_context("fork").Pool(processes=self.num_workers) as p:
            exp_col = p.starmap(
                predict_celltype_exp, tqdm(
                    zip(self.cell_ids, self.get_config_path, repeat(self.celltype_annot_dict), repeat(self.variants_rsid), repeat(self.genome), repeat(self.motif), repeat(self.inf_model)),
                    total=len(self.cell_ids)
                )
            )
        # exp_col = []
        # for cell_id in tqdm(self.cell_ids[:10]):
        #     exp_col.append(self.predict_celltype_exp(cell_id))
        return exp_col
    
    def generate_motif_diff(self, variants_file, save_motif_df=True):
        variants_ref = pd.read_csv(variants_file, sep='\t').set_index('ID').Ref.to_dict() 
        variants_alt = pd.read_csv(variants_file, sep='\t').set_index('ID').Alt.to_dict()
        variants_ld = pd.read_csv(variants_file, sep='\t')

        ld = {}
        lead_snp = ""
        for _, row in variants_ld.iterrows():
            if row['Variant/LD'] == 'variant':
                lead_snp = row['ID']
                ld[row['ID']] = row['ID']
            else:
                ld[row['ID']] = lead_snp
        
        variants_rsid = variants_rsid.df
        variants_rsid['Ref'] = variants_rsid.RSID.map(variants_ref)
        variants_rsid['Alt'] = variants_rsid.RSID.map(variants_alt)
        variants_rsid = self.variants_rsid.dropna()

        variants_rsid = Mutations(self.genome, variants_rsid)
        variants_rsid.collect_ref_sequence()
        variants_rsid.collect_alt_sequence()
        motif_diff = variants_rsid.get_motif_diff(self.motif)
        motif_diff_df = pd.DataFrame((motif_diff['Alt'].values-motif_diff['Ref'].values), index=variants_rsid.df.RSID.values, columns=motif.cluster_names)
        
        if save_motif_df:
            motif_diff_df.to_csv('motif_diff_df.csv')
        return ld, motif_diff_df
            
    def get_variant_score(self, motif_diff_score, variant, gene, cell):
        motif_importance = cell.get_gene_jacobian_summary(gene, 'motif')[0:-1]
        diff = motif_diff_score.copy().values
        diff[(diff<0) & (diff>-10)] = 0
        diff[(diff<0) & (diff<-10)] = -1
        diff[(diff>0) & (diff<10)] = 0
        diff[(diff>0) & (diff>10)] = 1
        
        combined_score = diff*motif_importance.values
        combined_score = pd.Series(combined_score, index=motif_diff_score.index.values).sort_values()
        combined_score = pd.DataFrame(combined_score, columns=['score'])
        combined_score['gene'] = gene
        combined_score['variant'] = variant.RSID
        try:
            combined_score['ld'] = self.ld[variant.RSID]
        except:
            combined_score['ld'] = variant.RSID
        combined_score['chrom'] = variant.Chromosome
        combined_score['pos'] = variant.Start
        combined_score['ref'] = variant.Ref
        combined_score['alt'] = variant.Alt
        combined_score['celltype'] = self.cell_type_annot_dict[cell.celltype]
        return combined_score

    @staticmethod
    def get_nearby_genes(variant, cell, distance=2000000):
        chrom = variant['Chromosome']
        pos = variant['Start']
        start = pos-distance
        end = pos+distance
        genes = cell.gene_annot.query('Chromosome==@chrom & Start>@start & Start<@end')
        return ','.join(np.unique(genes.gene_name.values))

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


if __name__=="__main__":
    cell_mut_col = CellMutCollection()
    results = cell_mut_col.predict_all_celltype_expression()
    breakpoint()
