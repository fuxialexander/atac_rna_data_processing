# This specify the code to
# 1. read variants from VCF files
# 2. read Structure variants from bedpe files
# 3. Manipulate the given sequence or bed files to accomodate the variants

import random
import time
import pandas as pd
import requests
from pysam import VariantFile
from tqdm import tqdm

from atac_rna_data_processing.io.region import GenomicRegionCollection
from atac_rna_data_processing.io.sequence import (DNASequence,
                                                  DNASequenceCollection)
import concurrent.futures
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np

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
        self.collect_ref_sequence(30)
        self.collect_alt_sequence(30)
        return

    def collect_ref_sequence(self, upstream=30, downstream=30):
        """Collect reference sequences centered at the mutation sites
        """
        self.Ref_seq = [s.seq for s in super().collect_sequence(
            upstream=upstream, downstream=downstream).sequences]

    def collect_alt_sequence(self, upstream=30, downstream=30):
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
