from pyranges import read_gtf
from pyranges import PyRanges as pr
import pandas as pd
import os
from ..gene import Gene


class Gencode(object):
    """Read gencode gene annotation given genome assembly and version, 
    returns a pandas dataframe"""

    def __init__(self, assembly="hg38", version=40):
        super(Gencode, self).__init__()

        self.assembly = assembly
        if self.assembly == "hg38":
            self.url = "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{version}/gencode.v{version}.annotation.gtf.gz".format(
                version=str(version))
            self.gtf = "gencode.v{version}.annotation.gtf.gz".format(
                version=str(version))
        elif self.assembly == "mm10":
            self.url = "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_{version}/gencode.v{version}.annotation.gtf.gz".format(
                version=str(version))
            self.gtf = "gencode.v{version}.annotation.gtf.gz".format(
                version=str(version))
        elif self.assembly == "hg19":
            self.url = "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{version}/gencode.v{version}lift37.annotation.gtf.gz".format(
                version=str(version))
            self.gtf = "gencode.v{version}lift37.annotation.gtf.gz".format(
                version=str(version))

        if os.path.exists("gencode.v{version}.{assembly}.feather".format(version=str(version), assembly=self.assembly)):
            self.gtf = pd.read_feather("gencode.v{version}.{assembly}.feather".format(
                version=str(version), assembly=self.assembly))
        else:
            if os.path.exists(self.gtf):
                self.gtf = read_gtf(self.gtf).as_df()
            else:
                # download gtf to current directory
                os.system("wget {url}".format(url=self.url))
                self.gtf = read_gtf(self.gtf).as_df()

            positive = self.gtf[(self.gtf.Feature == 'transcript') & (
                self.gtf.Strand == '+')][['Chromosome', 'Start', 'Start', 'Strand', 'gene_name', 'gene_id']]
            negative = self.gtf[(self.gtf.Feature == 'transcript') & (
                self.gtf.Strand == '-')][['Chromosome', 'End', 'End', 'Strand', 'gene_name', 'gene_id']]

            positive.columns = ['Chromosome', 'Start',
                                'End', 'Strand', 'gene_name', 'gene_id']
            negative.columns = ['Chromosome', 'Start',
                                'End', 'Strand', 'gene_name', 'gene_id']

            self.gtf = pd.concat([positive, negative],
                                 0).drop_duplicates().reset_index()
            self.gtf.to_feather("gencode.v{version}.{assembly}.feather".format(
                version=str(version), assembly=self.assembly))
        return

    def get_gene(self, gene_name):
        df = self.gtf[self.gtf.gene_name == gene_name]
        return Gene(name=df.gene_name.iloc[0], id=df.gene_id.iloc[0], chrom=df.Chromosome.iloc[0], strand=df.Strand.iloc[0], tss_list=pr(df[['Chromosome', 'Start', 'End']]))

    def get_gene_id(self, gene_id):
        df = self.gtf[self.gtf.gene_id.str.startswith(gene_id)]
        return Gene(name=df.gene_name.iloc[0], id=df.gene_id.iloc[0], chrom=df.Chromosome.iloc[0], strand=df.Strand.iloc[0], tss_list=pr(df[['Chromosome', 'Start', 'End']]))
