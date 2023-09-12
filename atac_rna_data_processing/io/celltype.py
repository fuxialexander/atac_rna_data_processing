import os

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nxcom
import numpy as np
import pandas as pd
import pkg_resources
import plotly.graph_objects as go
import seaborn as sns
import zarr
from atac_rna_data_processing.config.load_config import *
# import r2_score
from atac_rna_data_processing.io.causal_lib import (get_subnet, plot_comm,
                                                    plotly_networkx_digraph,
                                                    preprocess_net)
from atac_rna_data_processing.io.gene import TSS, Gene, GeneExp
from atac_rna_data_processing.io.motif import MotifClusterCollection
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
from plotly.subplots import make_subplots
from scipy.sparse import coo_matrix, csr_matrix, load_npz
from scipy.stats import zscore
from tqdm import tqdm

motif = NrMotifV1.load_from_pickle(
    pkg_resources.resource_filename("atac_rna_data_processing", "data/NrMotifV1.pkl")
)
motif_clusters = motif.cluster_names

# Load gencode_hg38 from feather file
gencode_hg38 = pd.read_feather(pkg_resources.resource_filename("atac_rna_data_processing", "data/gencode.v40.hg38.feather"))
gencode_hg38["Strand"] = gencode_hg38["Strand"].apply(lambda x: 0 if x == "+" else 1)
gene2strand = gencode_hg38.set_index("gene_name").Strand.to_dict()



class Celltype:
    """All information of a cell type"""

    def __init__(
        self,
        features: np.ndarray,
        num_region_per_sample: int,
        celltype: str,
        data_dir="../pretrain_human_bingren_shendure_apr2023",
        interpret_dir="Interpretation",
        input=False,
        jacob=False,
        embed=False,
        num_cls=2,
    ):
        self.celltype = celltype
        self.data_dir = data_dir
        self.interpret_dir = interpret_dir
        self.features = features
        self.num_features = features.shape[0]
        self.num_region_per_sample = num_region_per_sample
        self.focus = num_region_per_sample // 2
        self.num_cls = num_cls
        self.interpret_cell_dir = os.path.join(self.interpret_dir, celltype, "allgenes")
        self.gene_feather_path = f"{self.data_dir}/{celltype}.exp.feather"
        if os.path.exists(os.path.join(self.interpret_cell_dir, f"{self.celltype}.zarr")):
            self._zarr_data = zarr.open(
                os.path.join(self.interpret_cell_dir, f"{self.celltype}.zarr"), mode="a"
            )   
            self.genelist = self._zarr_data['avaliable_genes']
        else:
            self.genelist = np.load(
                os.path.join(self.interpret_cell_dir, "avaliable_genes.npy")
            )
        
        if not os.path.exists(self.gene_feather_path):
            self.gene_feather_path = (
                f"{self.interpret_dir}/{celltype}.gene_idx_dict.feather"
            )
        self.peak_annot = pd.read_csv(
            self.data_dir + celltype + ".csv", sep=","
        ).rename(columns={"Unnamed: 0": "index"})
        self.gene_annot = self.load_gene_annot()
        tss_idx = self.gene_annot.level_0.values
        self.tss_idx = tss_idx
        if input:
            self.input = load_npz(self.data_dir + celltype + ".watac.npz")[tss_idx]
            self.input_all = load_npz(self.data_dir + celltype + ".watac.npz")
            self.tss_accessibility = self.input[:, self.num_features - 1]
        self.gene_annot["Strand"] = self.gene_annot["gene_name"].apply(
                        lambda x: gene2strand[x])
        self.tss_strand = self.gene_annot.Strand.astype(int).values
        self.tss_start = self.peak_annot.iloc[tss_idx].Start.values
        


        if hasattr(self, "_zarr_data"):
            import time
            if jacob:
                start_time = time.time()
                self.jacobs = self._zarr_data['jacobians'][:]
                # print time with 2 decimals
                print(f'loaded jacobians in {time.time()-start_time:.2f} seconds')
            start_time = time.time()
            self.preds = np.array(self._zarr_data["preds"][:])
            self.preds = np.array(
                [
                    self.preds[i]
                    .reshape(self.num_region_per_sample, self.num_cls)[self.focus, j]
                    for i, j in enumerate(self.tss_strand)
                ]
            )
            self.obs = np.array(self._zarr_data["obs"][:])
            self.obs = np.array(
                [
                    self.obs[i]
                    .reshape(self.num_region_per_sample, self.num_cls)[self.focus, j]
                    for i, j in enumerate(self.tss_strand)
                ]
            )
            print(f'loaded preds and obs in {time.time()-start_time:.2f} seconds')
            if embed:
                start_time = time.time()
                self.embed = self._zarr_data["embeds_0"][:]
                print(f'loaded embeds in {time.time()-start_time:.2f} seconds')

        else:
            if embed:
                if os.path.exists(os.path.join(self.interpret_cell_dir, "embeds_0.npy")):
                    self.embed = np.load(
                        os.path.join(self.interpret_cell_dir, "embeds_0.npy")
                    )
                else:
                    raise ValueError("embeds_0.npy not found")

            if jacob:
                # check if os.path.join(self.interpret_cell_dir, "jacobians.zarr") exists, if not save the matrix to zarr file
                if os.path.exists(os.path.join(self.interpret_cell_dir, "jacobians.zarr")):
                    # load from zarr file
                    self.jacobs = zarr.open(
                        os.path.join(self.interpret_cell_dir, "jacobians.zarr"), mode="r"
                    )
                else:
                    jacob_npz = coo_matrix(
                        load_npz(os.path.join(self.interpret_cell_dir, "jacobians.npz"))
                    )
                    z = zarr.zeros(
                        shape=jacob_npz.shape,
                        chunks=(100, jacob_npz.shape[1]),
                        dtype=jacob_npz.dtype,
                    )
                    z.set_coordinate_selection(
                        (jacob_npz.row, jacob_npz.col), jacob_npz.data
                    )
                    # save to zarr file
                    zarr.save(os.path.join(self.interpret_cell_dir, "jacobians.zarr"), z)
                    self.jacobs = zarr.open(
                        os.path.join(self.interpret_cell_dir, "jacobians.zarr"), mode="r"
                    )

            self.preds = load_npz(os.path.join(self.interpret_cell_dir, "preds.npz"))
            self.preds = np.array(
                [
                    self.preds[i]
                    .toarray()
                    .reshape(self.num_region_per_sample, self.num_cls)[self.focus, j]
                    for i, j in enumerate(self.tss_strand)
                ]
            )
            self.obs = load_npz(os.path.join(self.interpret_cell_dir, "obs.npz"))
            self.obs = np.array(
                [
                    self.obs[i]
                    .toarray()
                    .reshape(self.num_region_per_sample, self.num_cls)[self.focus, j]
                    for i, j in enumerate(self.tss_strand)
                ]
            )


        self.gene_annot["pred"] = self.preds
        self.gene_annot["obs"] = self.obs
        if input:
            self.gene_annot["accessibility"] = self.tss_accessibility.toarray().reshape(
                -1
            )
        else:
            self.gene_annot["accessibility"] = 1
        self.gene_annot["Chromosome"] = self.peak_annot.iloc[tss_idx].Chromosome.values
        self.gene_annot["Start"] = self.tss_start
        self._gene_by_motif = None

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
        if not os.path.exists(self.gene_feather_path):
            print("Gene exp feather not found. Constructing gene annotation from gencode...")
            # construct gene annotation from gencode
            from pyranges import PyRanges as pr

            atac = pr(self.peak_annot, int64=True)
            # join the ATAC-seq data with the RNA-seq data
            exp = atac.join(
                pr(gencode_hg38, int64=True).extend(300), how="left"
            ).as_df()
            # save the data to feather file
            exp.reset_index(drop=True).to_feather(
                f"{self.data_dir}/{self.celltype}.exp.feather"
            )
            self.gene_feather_path = f"{self.data_dir}/{self.celltype}.exp.feather"
            gene_annot = exp
        else:
            print("Gene exp feather found. Loading...")
            gene_annot = pd.read_feather(self.gene_feather_path)
        if self.gene_feather_path.endswith(".exp.feather"):
            gene_annot = (
                gene_annot.groupby(["gene_name", "Strand"])["index"]
                .unique()
                .reset_index()
                .dropna()
                .query('gene_name!="-1"')
                .rename(columns={"index": "level_0"})
            )
        gene_annot = gene_annot.explode("level_0").reset_index(drop=True)
        gene_annot = gene_annot.iloc[self.genelist].reset_index(drop=True)
        return gene_annot

    def get_gene_idx(self, gene_name: str):
        """Get the index of a gene in the gene list."""
        return self.gene_annot[self.gene_annot["gene_name"] == gene_name].index.values

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
                gene_name,
                self.tss_idx[i],
                gene_chr,
                self.tss_start[i],
                self.tss_strand[i],
            )
            jacobs.append((self.get_tss_jacobian(self.jacobs[i], tss, multiply_input)))
        return jacobs

    def get_input_data(self, peak_id=None, focus=None, start=None, end=None):
        """Get input data from self.input_all using a slice"""
        # assert if all are None
        assert not all([peak_id, focus, start, end])
        if start is None:
            start = peak_id - focus
            end = peak_id + focus
        return self.input_all[start:end].toarray().reshape(-1, self.num_features)

    def get_tss_jacobian(self, jacob: np.ndarray, tss: TSS, multiply_input=True):
        """Get the jacobian of a TSS."""
        jacob = jacob.reshape(-1, self.num_region_per_sample, self.num_features)
        if multiply_input:
            input = self.get_input_data(peak_id=tss.peak_id, focus=self.focus)
            jacob = jacob * input
        region = tss.get_sample_from_peak(self.peak_annot, self.focus)
        tss_jacob = OneTSSJacobian(jacob, tss, region, self.features, self.num_cls)
        return tss_jacob

    def gene_jacobian_summary(self, gene, axis="motif", multiply_input=True):
        """
        Summarizes the Jacobian for a given gene.

        This function calculates the Jacobian for a given gene and summarizes it based on the specified axis. 
        If the axis is "motif", it concatenates the Jacobian summaries along axis 1 and then sums them. 
        If the axis is "region", it concatenates the Jacobian summaries along axis 0, groups them by index, chromosome, and start, 
        and then sums the scores.

        Parameters:
        gene (str): The gene for which the Jacobian is to be calculated.
        axis (str, optional): The axis along which to summarize the Jacobian. Defaults to "motif".
        multiply_input (bool, optional): If True, the input is multiplied. Defaults to True.

        Returns:
        pd.DataFrame: A DataFrame containing the summarized Jacobian.
        """
        gene_jacobs = self.get_gene_jacobian(gene, multiply_input)
        if axis == "motif":
            return pd.concat([jac.summarize(axis) for jac in gene_jacobs], axis=1).sum(
                axis=1
            )
        elif axis == "region":
            # concat in axis 0 and aggregate overlapping regions by sum the score and divided by number of tss
            return (
                pd.concat([j.summarize(axis="region") for j in gene_jacobs])
                .groupby(["index", "Chromosome", "Start"])
                .Score.sum()
                .reset_index()
            )
    
    @property
    def gene_by_motif(self):
        if self._gene_by_motif is None:
            self._gene_by_motif = self.get_gene_by_motif()
        return self._gene_by_motif
    
    @gene_by_motif.setter
    def gene_by_motif(self, value):
        self._gene_by_motif = value

    def get_gene_by_motif(self):
        """
        This method retrieves gene data by motif. It first checks if a zarr file exists for the cell type.
        If it does, it opens the zarr file and checks if 'gene_by_motif' is in the keys of the zarr data.
        If 'gene_by_motif' is found, it loads the data into a pandas DataFrame. If not, it computes the 
        jacobian for each gene and saves it to the zarr file.

        If a zarr file does not exist, it checks if a feather file exists. If it does, it loads the data 
        into a pandas DataFrame. If not, it computes the jacobian for each gene and saves it to a feather file.

        Finally, if the gene_by_motif data is a pandas DataFrame, it is converted to a GeneByMotif object. 
        If a zarr file exists, it checks if 'gene_by_motif_corr' is in the keys of the zarr data. If it is, 
        it loads the correlation data into the GeneByMotif object. If not, it computes the correlation and 
        saves it to the zarr file.

        Returns:
            GeneByMotif: An object that contains the gene data by motif and the correlation data.
        """
        if os.path.exists(os.path.join(self.interpret_cell_dir, f"{self.celltype}.zarr")):
            self._zarr_data = zarr.open(
                os.path.join(self.interpret_cell_dir, f"{self.celltype}.zarr"), mode="a"
            )
            if 'gene_by_motif' in self._zarr_data.keys():
                self._gene_by_motif = pd.DataFrame(
                    self._zarr_data["gene_by_motif"][:], columns=self.features)
            else:
                jacobs = []
                for g in tqdm(self.gene_annot.gene_name.unique()):
                    for j in self.get_gene_jacobian(g):
                        jacobs.append(j.motif_summary().T)
                jacobs_df = pd.concat(jacobs, axis=1).T
                # save to zarr
                self._zarr_data.array("gene_by_motif", jacobs_df.values, dtype="float32")
                self._gene_by_motif = jacobs_df
        elif os.path.exists(
            f"{self.interpret_dir}/{self.celltype}_gene_by_motif.feather"
        ):
            self._gene_by_motif = pd.read_feather(
                f"{self.interpret_dir}/{self.celltype}_gene_by_motif.feather"
            ).set_index("index")
        else:
            jacobs = []
            for g in tqdm(self.gene_annot.gene_name.unique()):
                for j in self.get_gene_jacobian(g):
                    jacobs.append(j.motif_summary().T)
            jacobs_df = pd.concat(jacobs, axis=1).T
            jacobs_df.reset_index().to_feather(
                f"{self.interpret_dir}/{self.celltype}_gene_by_motif.feather"
            )
            self._gene_by_motif = jacobs_df

        if isinstance(self._gene_by_motif, pd.DataFrame):
            self._gene_by_motif = GeneByMotif(self.celltype, self.interpret_dir, self.gene_by_motif)
            if os.path.exists(os.path.join(self.interpret_cell_dir, f"{self.celltype}.zarr")):
                if 'gene_by_motif_corr' in self._zarr_data.keys():
                    self._gene_by_motif.corr = pd.DataFrame(self._zarr_data["gene_by_motif_corr"][:], columns=self.features, index=self.features)
                    
                else:
                    # compute corr and save to zarr also
                    self._zarr_data.array("gene_by_motif_corr", self._gene_by_motif.get_corr().values, dtype="float32")

        return self._gene_by_motif

    def get_tf_pathway(self, tf, gp = None, quantile_cutoff=0.9, exp_cutoff=0, filter_str='term_size<1000 & term_size>500', significance_threshold_method='g_SCS'):
        """
        This function retrieves the pathway for a given transcription factor (tf) using g:Profiler.

        Parameters:
        tf (str): The transcription factor to get the pathway for.
        gp (GProfiler, optional): An instance of the GProfiler class. If None, a new instance will be created. Defaults to None.
        quantile_cutoff (float, optional): The quantile cutoff to use when selecting genes. Defaults to 0.9.
        exp_cutoff (int, optional): The expression cutoff to use when querying genes. Defaults to 0.
        filter_str (str, optional): The filter string to use when querying the g:Profiler results. Defaults to 'term_size<1000 & term_size>500'.
        significance_threshold_method (str, optional): The method to use for determining the significance threshold in g:Profiler. Defaults to 'g_SCS'.

        Returns:
        tuple: A tuple containing the filtered g:Profiler results and the unique genes in the pathways.
        """
        self.get_gene_by_motif()
        if gp is None:
            from gprofiler import GProfiler
            gp = GProfiler(return_dataframe=True)

        selected_index = (self.gene_by_motif.data[tf]>self.gene_by_motif.data[tf].quantile(quantile_cutoff))
        gene_list = self.gene_annot.loc[selected_index].query('pred>@exp_cutoff').gene_name.unique()
        go = gp.profile(organism='hsapiens', query=list(gene_list), user_threshold=0.05, no_evidences=False, significance_threshold_method=significance_threshold_method)
        go_filtered = go.query(filter_str)
        pathway_genes = np.unique(np.concatenate(go_filtered.intersections.values))
        return go_filtered, pathway_genes
    
    def get_highest_exp_genes(self, genes):
        '''
        This code takes in a list of genes and returns the gene with the highest expression value. 
        '''
        return self.gene_annot.query(f'gene_name.isin(@genes)').sort_values('pred', ascending=False).head(1).gene_name.values[0]

    def get_genes_exp(self, genes):
        return self.gene_annot.query(f'gene_name.isin(@genes)').sort_values('pred', ascending=False)
    
    def get_tf_exp_str(self, motif, m):
        """
        This method generates a formatted string of gene names and their corresponding predicted expression values.
        The expression values are averaged and sorted in descending order.

        Parameters:
        motif (Motif object): The motif object containing the cluster gene list.
        m (int): The index to access the specific cluster gene list from the motif object.

        Returns:
        str: A string representation of gene names and their corresponding predicted expression values.
            The string is formatted as 'gene_name\tpred', where 'pred' is a 2 digit floating point number.
            Each gene name and its corresponding predicted expression value are separated by a '<br />'.

        Example:
        'gene1\t1.23<br />gene2\t0.56<br />gene3\t0.45'
        """
        if m not in motif.cluster_gene_list.keys():
            return m
        motif_cluster_genes = motif.cluster_gene_list[m]
        motif_cluster_genes_exp = self.get_genes_exp(motif_cluster_genes).groupby('gene_name').pred.mean().sort_values(ascending=False)
        # turn in to a formated table in one string: gene_name\tpred, 2 digit floating point
        return '<br />'.join([f'{gene_name}\t{pred:.2f}' for gene_name, pred in motif_cluster_genes_exp.items()])

    def get_tf_exp_mean(self, motif, m):
        """
        Calculate the mean expression of transcription factors (TFs) for a given motif and cluster.

        Parameters:
        motif (Motif): The motif object containing information about the motif and associated genes.
        m (int): The cluster index for which the mean TF expression is to be calculated.

        Returns:
        float: The mean expression of TFs for the given motif and cluster.
        """
        if m not in motif.cluster_gene_list.keys():
            return np.nan
        motif_cluster_genes = motif.cluster_gene_list[m]
        motif_cluster_genes_exp = self.get_genes_exp(motif_cluster_genes).groupby('gene_name').pred.mean()
        return motif_cluster_genes_exp.mean()

    def get_gene_pred(self, gene_name: str):
        """Get the prediction of a gene."""
        return self.preds[self.get_gene_idx(gene_name)]

    def get_gene_obs(self, gene_name: str):
        """Get the observed value of a gene."""
        return self.obs[self.get_gene_idx(gene_name)]

    def get_gene_annot(self, gene_name: str):
        """Get the gene annotation of a gene."""
        return self.gene_annot[self.gene_annot["gene_name"] == gene_name]

    def get_gene_accessibility(self, gene_name: str):
        """Get the accessibility of a gene."""
        gene_idx = self.get_gene_idx(gene_name)
        return (
            self.tss_accessibility[gene_idx]
            if hasattr(self, "tss_accessibility")
            else None
        )

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
        return [
            TSS(
                gene_name,
                self.tss_idx[i],
                gene_chr,
                self.tss_start[i],
                self.tss_strand[i],
            )
            for i in gene_idx
        ]

    def get_gene_chromosome(self, gene_name: str):
        """Get the chromosome of a gene."""
        return self.gene_annot[self.gene_annot["gene_name"] == gene_name][
            "Chromosome"
        ].values[0]

    def get_gene_jacobian_summary(self, gene_name: str, axis="motif"):
        """Get the jacobian summary of a gene."""
        gene_jacobs = self.get_gene_jacobian(gene_name)
        if axis == "motif":
            return pd.concat([j.summarize(axis) for j in gene_jacobs], axis=1).sum(
                axis=1
            )
        elif axis == "region":
            # concat in axis 0 and aggregate overlapping regions by sum the score and divided by number of tss
            return (
                pd.concat([j.summarize(axis="region") for j in gene_jacobs])
                .groupby(["index", "Chromosome", "Start"])
                .Score.sum()
                .reset_index()
            )

    def plot_gene_motifs(self, gene, motif, overwrite=False):
        r = self.get_gene_jacobian_summary(gene, 'motif')
        m = r.sort_values(ascending=False).head(10).index.values
        fig, ax = plt.subplots(2, 5, figsize=(10, 4), sharex=False, sharey=False)
        for i, m_i in enumerate(m):
            if not os.path.exists(f'assets/{m_i.replace("/", "_")}.png') or overwrite==True:
                motif.get_motif_cluster_by_name(m_i).seed_motif.plot_logo(filename=f'assets/{m_i.replace("/", "_")}.png', logo_title='', size='medium', ic_scale=True)
            # show logo in ax[i] from the png file
            img = plt.imread(f'assets/{m_i.replace("/", "_")}.png')
            ax[i//5][i%5].imshow(img)
            ax[i//5][i%5].axis('off')
            # add title to highest expressed gene
            if m_i in motif.cluster_gene_list.keys():
                motif_cluster_genes = motif.cluster_gene_list[m_i]
            if len(motif_cluster_genes) > 1:
                ax[i//5][i%5].set_title(f'{m_i}:{self.get_highest_exp_genes(motif_cluster_genes)}')
            else:
                ax[i//5][i%5].set_title(f'{m_i}')
            
        return fig, ax
    
    def plotly_motif_subnet(self, motif, m, type='neighbors', threshold='auto'):
        """
        Plots a subnet of motifs.

        This function generates a subnet of motifs based on the given parameters and plots it using plotly. 
        The subnet is preprocessed and the TF expression string and mean are calculated for each motif in the cluster gene list.

        Parameters:
        motif (Motif): The motif object to plot.
        m (int): The motif index to plot.
        type (str, optional): The type of subnet to generate. Can be 'neighbors', 'parents', or 'children'. Defaults to 'neighbors'.
        threshold (str or float, optional): The threshold for preprocessing the network. Can be 'auto' or a float. If 'auto', the threshold is set to the 70th percentile of the absolute weight values. Defaults to 'auto'.

        Returns:
        plotly.graph_objs._figure.Figure: The plotly figure object of the plotted subnet.
        """
        causal = self.gene_by_motif.get_causal()
        if threshold == 'auto':
            threshold = pd.DataFrame(causal.edges(data='weight'), columns=['From', 'To', 'Weight']).sort_values('Weight').Weight.abs().quantile(0.7)
        subnet = preprocess_net(causal.copy(), threshold)
        subnet = get_subnet(subnet, m, type)
        tf_exp_str = {m:self.get_tf_exp_str(motif, m) for m in motif.cluster_names}
        tf_exp_mean = {m:self.get_tf_exp_mean(motif, m) for m in motif.cluster_names}
        return plotly_networkx_digraph(subnet, tf_exp_str, tf_exp_mean)

    def plotly_gene_exp(self):
        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame(self.gene_annot)
        fig = px.scatter(df.groupby('gene_name')[['obs', 'pred','accessibility']].mean().reset_index(), x='obs', y='pred', color='accessibility', hover_name='gene_name', 
                        labels={'obs': 'Observed log10 TPM', 'pred': 'Predicted log10 TPM', 'accessibility': 'TSS Accessibility'},
                        template='plotly_white', width=800, height=700, opacity=0.5, marginal_x='histogram', marginal_y='histogram')
        # add a text annotation of pearson correlation
        fig.add_annotation(
            x=0.1,
            y=1.0,
            text=f"Cell type: {self.celltype_name}<br />Pearson correlation: {df.groupby('gene_name')[['obs', 'pred']].mean().corr().values[0,1]:.2f}",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(
                family="Arial",
                size=16,
                color="black"
            ),
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )
        # all font to Arial
        fig.update_layout(
            font_family="Arial",
            font_color="black",
            title_font_family="Arial",
            title_font_color="black",
            legend_title_font_color="black",
            legend_font_color="black",
            xaxis_title_font_family="Arial",
            yaxis_title_font_family="Arial",
            xaxis_title_font_color="black",
            yaxis_title_font_color="black",
            xaxis_tickfont_family="Arial",
            yaxis_tickfont_family="Arial",
            xaxis_tickfont_color="black",
            yaxis_tickfont_color="black",
            xaxis_tickcolor="black",
            yaxis_tickcolor="black")

        return fig

    def plot_gene_regions(self, gene, plotly=False):
        r = self.get_gene_jacobian_summary(gene, 'region')
        js = self.get_gene_jacobian(gene)
        r['End'] = self.peak_annot.iloc[r['index'].values].End.values
        r = r[['index', 'Chromosome', 'Start', 'End', 'Score']]
        r_motif = pd.concat([j.data for j in js],axis=0).drop(['Chromosome', 'Start', 'End'], axis=1).groupby('index').mean()
        r = r.merge(r_motif, left_on='index', right_index=True)
        if plotly:
            return self.plot_region_plotly(r)
        else:
            return self.plot_region(r)

    def plot_region(df):
        # plot the region using rectangles defined by start and end and height defined by score
        # df: dataframe with columns ['Start', 'End', 'Score']
        # return: plot
        df = df.sort_values('Score', ascending=False)
        df['Height'] = df.Score.abs() / df.Score.abs().max()
        df['Width'] = df.End - df.Start
        df['X'] = df.Start
        df['Y'] = df.Height

        fig, ax = plt.subplots(figsize=(10, 2))
        for i, row in df.iterrows():
            ax.add_patch(plt.Rectangle((row.X, 0), row.Width, row.Height, color='red'))
        ax.set_xlim(df.Start.min(), df.End.max())
        # add y=0
        ax.plot([df.Start.min(), df.End.max()], [0, 0], color='black')
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel(f'Genomic Position on Chromosome {df.Chromosome.iloc[0][3:]}')
        # remove top and right spines
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
        return fig, ax

    def plot_region_plotly(self, df: pd.DataFrame) -> go.Figure:
        # Create a subplot with 2 vertical panels; the second panel will be used for gene annotations
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.8, 0.2])
        
        hover_text = df.apply(lambda row: f"{row['Chromosome']}:{row['Start']}-{row['End']}<br />Top 5 motifs: {', '.join(row.iloc[5:].sort_values()[-5:].index.values)}<br />Bottom 5 motifs: {', '.join(row.iloc[5:].sort_values()[:5].index.values)}", axis=1)
        df['HoverText'] = hover_text

        # Process main DataFrame to sort by 'Score' and normalize height
        sorted_df = df.sort_values('Score', ascending=False)
        max_score = sorted_df['Score'].abs().max()
        sorted_df['NormalizedHeight'] = sorted_df['Score'].abs() / max_score
        
        # Compute genomic span, normalized positions and widths
        genomic_span = sorted_df['End'].max() - sorted_df['Start'].min()
        x_positions = sorted_df['Start']
        widths = (sorted_df['End'] - sorted_df['Start']) / genomic_span
        heights = sorted_df['NormalizedHeight']
        
        # Prepare hover text for main panel
        # Add bar trace for main genomic data
        fig.add_trace(
            go.Bar(
                x=x_positions,
                y=heights,
                width=widths,
                orientation='v',
                text=hover_text,
                hoverinfo='text'
            ),
            row=1, col=1
        )
        # add a scatter trace for each bar for top 10 % positions
        top10 = sorted_df.sort_values('NormalizedHeight', ascending=False).head(int(len(sorted_df)*0.1))

        fig.add_trace(
            go.Scatter(
                x=top10['Start'],
                y=top10['NormalizedHeight'],
                mode='markers',
                marker=dict(color=top10['NormalizedHeight'], colorscale='plotly3'),
                text=top10['HoverText'],
                hoverinfo='text'
            ),
            row=1, col=1
        )
        # Query gene annotations for the same chromosome and genomic range
        gene_start = sorted_df['Start'].min()
        gene_end = sorted_df['End'].max()
        chromosome = sorted_df.iloc[0].Chromosome
        genes_to_show = self.gene_annot.query(f'Chromosome==@chromosome and Start>=@gene_start and Start<=@gene_end')
        
        # Add scatter trace for gene annotations
        fig.add_trace(
            go.Scatter(
                x=genes_to_show['Start'],
                y=genes_to_show['Strand'],
                mode='markers',
                marker=dict(color=genes_to_show['Strand'], colorscale='spectral'),
                text=genes_to_show['gene_name'],
                hoverinfo='text'
            ),
            row=2, col=1
        )

        # set y-axis ticks for gene annotations (0: '+', 1: '-')
        fig.update_yaxes(row=2, col=1, tickmode='array', tickvals=[0, 1], ticktext=['+', '-'])
        
        # Update layout
        chrom_id = sorted_df.iloc[0]['Chromosome'][3:]
        fig.update_layout(
            xaxis=dict(range=[sorted_df['Start'].min(), sorted_df['End'].max()], title=f'Genomic Position on Chromosome {chrom_id}'),
            yaxis=dict(range=[0, 1.2], showticklabels=False),
            showlegend=False,
            plot_bgcolor='white'
        )

        return fig

class GETCellType(Celltype):
    def __init__(self, celltype, config):
        features = config.celltype.features
        if features == "NrMotifV1":
            features = np.array(motif_clusters + ["Accessibility"])
        num_region_per_sample = config.celltype.num_region_per_sample
        data_dir = config.celltype.data_dir
        interpret_dir = config.celltype.interpret_dir
        input = config.celltype.input
        jacob = config.celltype.jacob
        embed = config.celltype.embed
        num_cls = config.celltype.num_cls
        super().__init__(
            features,
            num_region_per_sample,
            celltype,
            data_dir,
            interpret_dir,
            input,
            jacob,
            embed,
            num_cls,
        )


class OneTSSJacobian:
    """Jacobian data for one TSS."""

    def __init__(
        self,
        data: np.ndarray,
        tss: TSS,
        region: pd.DataFrame,
        features: list,
        num_cls=2,
        num_region_per_sample=200,
        num_features=283,
    ) -> None:
        # check if the data dimension is correct:
        # (num_cls, num_region_per_sample, num_features)
        assert (
            data.reshape(-1).shape[0] == num_cls * num_region_per_sample * num_features
        )
        self.TSS = tss
        data = data.reshape(num_cls, num_region_per_sample, num_features)[tss.strand]
        data_df = pd.DataFrame(data, columns=features)
        data_df = pd.concat(
            [region.reset_index(drop=True), data_df.reset_index(drop=True)],
            axis=1,
            ignore_index=True,
        )
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

    def motif_summary(self, stats="mean"):
        """Summarize the motif scores."""
        # assert stats in ['mean', 'max']
        motif_data = self.data[self.features]
        if stats == "mean":
            return motif_data.mean(axis=0)
        elif stats == "max":
            return motif_data.max(axis=0)
        elif stats == "absmean":
            return motif_data.abs().mean(axis=0)
        elif stats == "signed_absmean":
            return motif_data.abs().mean(axis=0) * np.sign(motif_data.mean(axis=0))
        # if stats is a function
        elif callable(stats):
            return motif_data.apply(stats, axis=0)

    def region_summary(self, stats="mean"):
        """Summarize the motif scores."""
        data = self.data[self.features]
        if stats == "mean":
            region_data = data.mean(axis=1)
        elif stats == "max":
            region_data = data.max(axis=1)
        elif stats == "absmean":
            region_data = data.abs().mean(axis=1)
        # if stats is a function
        elif callable(stats):
            region_data = data.apply(stats, axis=1)
        data = self.data.iloc[:, 0:3]
        data["Score"] = region_data
        return data

    def summarize(self, axis="motif", stats="mean"):
        """Summarize the data."""
        if axis == "motif":
            return self.motif_summary(stats)
        elif axis == "region":
            return self.region_summary(stats)

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
        focus = self.num_region_per_sample // 2
        start_idx = self.TSS.peak_id - focus
        gene = self.TSS.gene_name
        # get a dataframe of {'Chromosome', 'Start', 'End', 'Gene', 'Strand', 'Start', 'Pred', 'Accessibility', 'Motif', 'Score'}
        df = self.peak_annot.iloc[
            start_idx : start_idx + self.num_region_per_sample
        ].copy()[["Chromosome", "Start", "End"]]
        df = df.iloc[rows]
        df["Motif"] = [self.data.columns[m] for m in cols]
        df["Score"] = values
        df["Gene"] = gene
        df["Strand"] = self.TSS.strand
        df["TSS"] = self.TSS.start
        df = df[
            ["Chromosome", "Start", "End", "Gene", "Strand", "TSS", "Motif", "Score"]
        ]
        return df


class GeneByMotif(object):
    """Gene by motif jacobian data."""

    def __init__(self, celltype, interpret_dir, jacob) -> None:
        self.celltype = celltype
        self.data = jacob
        self.interpret_dir = interpret_dir
        self._corr = None
        self._causal = None
    
    def __repr__(self) -> str:
        return f"""Celltype: {self.celltype}
        Jacob shape: {self.data.shape}
        """

    @property
    def corr(self):
        """Get the correlation."""
        if self._corr is None:
            self._corr = self.get_corr()
        return self._corr

    @corr.setter
    def corr(self, value):
        self._corr = value
    
    def get_corr(self, method="spearman", diagal_to_zero=True):
        """Get the motif correlation."""
        corr = self.data.corr(method=method)
        if diagal_to_zero:
            corr = self.set_diagnal_to_zero(corr)
        return corr

    @property
    def causal(self):
        """Get the causal graph."""
        if self._causal is None:
            self._causal = self.get_causal()
        return self._causal
    
    @causal.setter
    def causal(self, value):
        self._causal = value
            
    def get_causal(self, edgelist_file=None, permute_columns=True, n=3, overwrite=False):
        if edgelist_file is not None and os.path.exists(edgelist_file) and not overwrite:
            return nx.read_weighted_edgelist(edgelist_file, create_using=nx.DiGraph)
        
        zarr_data_path = os.path.join(self.interpret_dir, self.celltype, "allgenes", f"{self.celltype}.zarr")
        
        if os.path.exists(zarr_data_path+"/causal") and not overwrite:
            return self.load_causal_from_zarr(zarr_data_path)
        
        data = zscore(self.data, axis=0)
        
        zarr_data = zarr.open(zarr_data_path, mode="a")
        
        for i in tqdm(range(n)):
            if not f"causal_{i}" in zarr_data.keys() or overwrite:
                permuted_data = data.iloc[:, np.random.permutation(data.shape[1])] if permute_columns else data.copy()
                causal_g = self.create_causal_graph(permuted_data)
                self.save_causal_to_zarr(zarr_data, causal_g, i)

        # Load all n numpy arrays and compute the average
        average_causal = self.compute_average_causal(zarr_data, n)

        # Save the average to 'causal' in zarr
        zarr_data.array("causal", average_causal, dtype="float32", overwrite=True)
        
        # create networkx graph from average_causal
        causal_g = nx.from_numpy_array(average_causal, create_using=nx.DiGraph)
        causal_g = nx.relabel_nodes(causal_g, dict(zip(range(len(self.data.columns)), self.data.columns)))
                
        if edgelist_file:
            nx.write_weighted_edgelist(causal_g, edgelist_file)

        return causal_g

    def load_causal_from_zarr(self, zarr_data_path):
        zarr_data = zarr.open(zarr_data_path, mode="a")
        causal_g_numpy = zarr_data["causal"][:]
        causal_g = nx.from_numpy_array(causal_g_numpy, create_using=nx.DiGraph)
        causal_g = nx.relabel_nodes(causal_g, dict(zip(range(len(self.data.columns)), self.data.columns)))
        return causal_g

    def create_causal_graph(self, data):
        try:
            import cdt
        except ImportError:
            print("Please install cdt package to use this function.")
            return None
        model = cdt.causality.graph.LiNGAM()
        output = model.predict(data)
        causal_g = preprocess_net(output.copy())
        causal_g_numpy = nx.to_numpy_array(causal_g, dtype="float32", nodelist=self.data.columns)
        causal_g = nx.from_numpy_array(causal_g_numpy, create_using=nx.DiGraph)
        causal_g = nx.relabel_nodes(causal_g, dict(zip(range(len(self.data.columns)), self.data.columns)))
        return causal_g

    def save_causal_to_zarr(self, zarr_data, causal_g, index):
        causal_g_numpy = nx.to_numpy_array(causal_g, dtype="float32", nodelist=self.data.columns)
        zarr_data.array(f"causal_{index}", causal_g_numpy, dtype="float32", overwrite=True)
    
    def compute_average_causal(self, zarr_data, n):
        causal_arrays = [zarr_data[f"causal_{i}"][:] for i in range(n)]
        average_causal = np.mean(causal_arrays, axis=0)
        return average_causal

    def set_diagnal_to_zero(self, df: pd.DataFrame):
        for i in range(df.shape[0]):
            df.iloc[i, i] = 0
        return df

