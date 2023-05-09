from ast import Expression
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import norm
import pkg_resources
import os
from atac_rna_data_processing.io.gene import Gene, TSS, GeneExp
import seaborn as sns
from atac_rna_data_processing.io.motif import MotifClusterCollection
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
from atac_rna_data_processing.config.load_config import *
motif = NrMotifV1.load_from_pickle(pkg_resources.resource_filename('atac_rna_data_processing', 'data/NrMotifV1.pkl'))
# cannot find ../data/NrMotifV1.pkl, fix it using 

# Load motif_annot and motif_clusters from tsv and txt files
# motif_annot = pd.read_csv("data/motif_annot.tsv", sep='\t', names=[
#                           'cluter_id', 'cluster_name', 'motif', 'symbol', 'db', 'seq', 'Relative_orientation', 'Width', 'Left_offset', 'Right_offset'])

motif_clusters = motif.cluster_names

# Load gencode_hg38 from feather file
gencode_hg38 = pd.read_feather("./gencode.v40.hg38.feather")
gencode_hg38['Strand'] = gencode_hg38['Strand'].apply(
    lambda x: 0 if x == '+' else 1)
gene2strand = gencode_hg38.set_index('gene_name').Strand.to_dict()


class Celltype:
    """All information of a cell type"""

    def __init__(self, features: np.ndarray, num_region_per_sample: int, 
                 celltype: str, data_dir="../pretrain_human_bingren_shendure_apr2023",
                 interpret_dir="Interpretation", input=False, jacob=False):
        self.celltype = celltype
        self.data_dir = data_dir
        self.interpret_dir = interpret_dir
        self.features = features
        self.num_features = features.shape[0]
        self.num_region_per_sample = num_region_per_sample
        self.focus = num_region_per_sample // 2
        self.interpret_cell_dir = os.path.join(
            self.interpret_dir, celltype, "allgenes")
        self.gene_feather_path = f'{self.data_dir}/{celltype}.exp.feather'
        if not os.path.exists(self.gene_feather_path):
            self.gene_feather_path = f'{self.interpret_dir}/{celltype}.gene_idx_dict.feather'
        self.genelist = np.load(os.path.join(
            self.interpret_cell_dir, 'avaliable_genes.npy'))
        self.gene_annot = self.load_gene_annot()
        self.gene_annot['Strand'] = self.gene_annot['gene_name'].apply(
            lambda x: gene2strand[x])
        self.peak_annot = pd.read_csv(
            self.data_dir + celltype + ".csv", sep=',').rename(columns={'Unnamed: 0': 'index'})
        tss_idx = self.gene_annot.level_0.values
        self.tss_idx = tss_idx
        if input:
            self.input = load_npz(
                self.data_dir + celltype + ".watac.npz")[tss_idx]
            self.input_all = load_npz(self.data_dir + celltype + ".watac.npz")
            self.tss_accessibility = self.input[:, self.num_features - 1]
        self.tss_strand = self.gene_annot.Strand.values
        self.tss_start = self.peak_annot.iloc[tss_idx].Start.values
        if jacob:
            self.jacobs = load_npz(os.path.join(
                self.interpret_cell_dir, "jacobians.npz"))
        self.preds = load_npz(os.path.join(
            self.interpret_cell_dir, "preds.npz"))
        self.preds = np.array([self.preds[i].toarray().reshape(self.num_region_per_sample, 2)[
                              self.focus, j] for i, j in enumerate(self.tss_strand)])
        self.obs = load_npz(os.path.join(self.interpret_cell_dir, "obs.npz"))
        self.obs = np.array([self.obs[i].toarray().reshape(self.num_region_per_sample, 2)[
                            self.focus, j] for i, j in enumerate(self.tss_strand)])
        self.gene_annot['pred'] = self.preds
        self.gene_annot['obs'] = self.obs
        if input:
            self.gene_annot['accessibility'] = self.tss_accessibility.toarray(
            ).reshape(-1)
        else:
            self.gene_annot['accessibility'] = 1
        self.gene_annot['Chromosome'] = self.peak_annot.iloc[tss_idx].Chromosome.values
        self.gene_annot['Start'] = self.tss_start

    def __repr__(self) -> str:
        return f"""Celltype: {self.celltype}
        Data dir: {self.data_dir}
        Interpretation dir: {self.interpret_dir}
        Number of regions per sample: {self.num_region_per_sample}
        Number of features: {self.num_features}
        Number of genes: {self.gene_annot.gene_name.unique().shape[0]}
        Number of TSS: {self.tss_idx.shape[0]}
        Number of peaks: {self.peak_annot.shape[0]}
        """

    def load_gene_annot(self):
        """Load gene annotations from feather file."""
        gene_annot = pd.read_feather(self.gene_feather_path)
        if self.gene_feather_path.endswith(".exp.feather"):
            gene_annot = gene_annot.groupby(['gene_name', 'Strand'])['index'].unique(
            ).reset_index().dropna().query('gene_name!="-1"').rename(columns={'index': 'level_0'})
        gene_annot = gene_annot.explode('level_0').reset_index(drop=True)
        gene_annot = gene_annot.iloc[self.genelist].reset_index(drop=True)
        return gene_annot

    def get_gene_idx(self, gene_name: str):
        """Get the index of a gene in the gene list."""
        return self.gene_annot[self.gene_annot['gene_name'] == gene_name].index.values

    def get_tss_idx(self, gene_name: str):
        """Get the TSS index of a gene."""
        return self.tss_idx[self.get_gene_idx(gene_name)]

    def get_gene_jacobian(self, gene_name: str, multiply_input=True):
        """Get the jacobian of a gene."""
        gene_idx = self.get_gene_idx(gene_name)
        gene_chr = self.get_gene_chromosome(gene_name)
        jacobs = []
        for i in gene_idx:
            # get a TSS object
            tss = TSS(
                gene_name, self.tss_idx[i], gene_chr, self.tss_start[i], self.tss_strand[i])
            jacobs.append((self.get_tss_jacobian(self.jacobs[i], tss, multiply_input)))
        return jacobs

    def get_input_data(self, peak_id=None, focus=None, start=None, end=None):
        """Get input data from self.input_all using a slice"""
        # assert if all are None
        assert not all([peak_id, focus, start, end])
        if start is None:
            start = peak_id - focus
            end = peak_id + focus
        return self.input_all[start:end].toarray().reshape(-1, self.num_region_per_sample, self.num_features)

    def get_tss_jacobian(self, jacob: np.ndarray, tss: TSS, multiply_input=True):
        """Get the jacobian of a TSS."""
        jacob = jacob.toarray().reshape(-1, self.num_region_per_sample, self.num_features)
        if multiply_input:
            input = self.get_input_data(peak_id=tss.peak_id, focus=self.focus)
            jacob = jacob * input
        region = tss.get_sample_from_peak(self.peak_annot, self.focus)
        tss_jacob = OneTSSJacobian(jacob, tss, region, self.features)
        return tss_jacob

    def summarize_gene_jacobian(self, jacob: np.ndarray):
        """Summarize the jacobian of a gene."""
        return np.sum(jacob, axis=0)

    def get_gene_pred(self, gene_name: str):
        """Get the prediction of a gene."""
        return self.preds[self.get_gene_idx(gene_name)]

    def get_gene_obs(self, gene_name: str):
        """Get the observed value of a gene."""
        return self.obs[self.get_gene_idx(gene_name)]

    def get_gene_annot(self, gene_name: str):
        """Get the gene annotation of a gene."""
        return self.gene_annot[self.gene_annot['gene_name'] == gene_name]

    def get_gene_accessibility(self, gene_name: str):
        """Get the accessibility of a gene."""
        gene_idx = self.get_gene_idx(gene_name)
        return self.tss_accessibility[gene_idx] if hasattr(self, 'tss_accessibility') else None

    def get_gene_strand(self, gene_name: str):
        """Get the strand of a gene."""
        return self.tss_strand[self.get_gene_idx(gene_name)].unique()[0]

    def get_gene_tss_start(self, gene_name: str):
        """Get the start position of a gene."""
        return self.tss_start[self.get_gene_idx(gene_name)]

    def get_gene_tss(self, gene_name: str):
        """Get the TSS objects of a gene."""
        gene_idx = self.get_gene_idx(gene_name)
        gene_chr = self.get_gene_chromosome(gene_name)
        return [TSS(gene_name, self.tss_idx[i], gene_chr, self.tss_start[i], self.tss_strand[i]) for i in gene_idx]

    def get_gene_chromosome(self, gene_name: str):
        """Get the chromosome of a gene."""
        return self.gene_annot[self.gene_annot['gene_name'] == gene_name]['Chromosome'].values[0]

class GETCellType(Celltype):
    def __init__(self, celltype, config):
        features = config.celltype.features
        if features == 'NrMotifV1':
            features = np.array(motif_clusters+['Accessibility'])
        num_region_per_sample = config.celltype.num_region_per_sample
        data_dir=config.celltype.data_dir
        interpret_dir=config.celltype.interpret_dir
        input=config.celltype.input
        jacob=config.celltype.jacob
        super().__init__(features, num_region_per_sample, celltype, data_dir, interpret_dir, input, jacob)

    

class OneTSSJacobian():
    """Jacobian data for one TSS."""

    def __init__(self, data: np.ndarray, tss: TSS, region: pd.DataFrame, features: list, num_cls=2, num_region_per_sample=200, num_features=283) -> None:
        # check if the data dimension is correct:
        # (num_cls, num_region_per_sample, num_features)
        assert data.reshape(-1).shape[0] == num_cls * \
            num_region_per_sample * num_features
        self.TSS = tss
        data = data.reshape(num_cls, num_region_per_sample,
                            num_features)[tss.strand]
        data_df = pd.DataFrame(data, columns=features)
        data_df = pd.concat([region.reset_index(drop=True), data_df.reset_index(drop=True)], axis=1, ignore_index=True)
        data_df.columns = region.columns.tolist() + list(features)
        self.data = data_df
        self.num_cls = num_cls
        self.features = features
        self.num_region_per_sample = num_region_per_sample
        self.num_features = num_features

    def __repr__(self) -> str:
        return f"""TSS: {self.TSS}
        Data shape: {self.data.shape}
        """

    # function for arbitrary transformation of the data

    def transform(self, func):
        """Transform the data."""
        self.data = func(self.data)
        return self.data

    def motif_summary(self, stats='mean'):
        """Summarize the motif scores."""
        # assert stats in ['mean', 'max']
        motif_data = self.data[self.features]
        if stats == 'mean':
            return motif_data.mean(axis=0)
        elif stats == 'max':
            return motif_data.max(axis=0)
        elif stats == 'absmean':
            return motif_data.abs().mean(axis=0)
        elif stats == 'signed_absmean':
            return motif_data.abs().mean(axis=0) * np.sign(motif_data.mean(axis=0))
        # if stats is a function
        elif callable(stats):
            return motif_data.apply(stats, axis=0)

    def region_summary(self, stats='mean'):
        """Summarize the motif scores."""
        data = self.data[self.features]
        if stats == 'mean':
            region_data = data.mean(axis=1)
        elif stats == 'max':
            region_data = data.max(axis=1)
        elif stats == 'absmean':
            region_data = data.abs().mean(axis=1)
        # if stats is a function
        elif callable(stats):
            region_data = data.apply(stats, axis=1)
        data = self.data.iloc[:,0:3]
        data['Score'] = region_data
        return data
        

    def get_pairs_with_l2_cutoff(self, cutoff: float):
        """Get the pairs with L2 Norm cutoff."""
        l2_norm = np.linalg.norm(self.data, axis=1)
        if l2_norm == 0:
            return None
        v = self.data.values
        v[v**2 < cutoff] = 0
        v = csr_matrix(v)
        # Get the row, col, and value arrays from the csr_matrix
        rows, cols = v.nonzero()  # row are region idx, col are motif/feature idx
        values = v.data
        focus = self.num_region_per_sample//2
        start_idx = self.TSS.peak_id-focus
        gene = self.TSS.gene_name
        # get a dataframe of {'Chromosome', 'Start', 'End', 'Gene', 'Strand', 'Start', 'Pred', 'Accessibility', 'Motif', 'Score'}
        df = self.peak_annot.iloc[start_idx:start_idx +
                                  self.num_region_per_sample].copy()[['Chromosome', 'Start', 'End']]
        df = df.iloc[rows]
        df['Motif'] = [self.data.columns[m] for m in cols]
        df['Score'] = values
        df['Gene'] = gene
        df['Strand'] = self.TSS.strand
        df['TSS'] = self.TSS.start
        df = df[['Chromosome', 'Start', 'End', 'Gene',
                 'Strand', 'TSS', 'Motif', 'Score']]
        return df
