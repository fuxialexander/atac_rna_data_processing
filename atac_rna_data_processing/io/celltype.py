import os

import numpy as np
import pandas as pd
import pkg_resources
import zarr
from scipy.sparse import csr_matrix, load_npz, coo_matrix
from tqdm import tqdm
import zarr
from atac_rna_data_processing.config.load_config import *
from atac_rna_data_processing.io.gene import TSS, Gene, GeneExp
from atac_rna_data_processing.io.motif import MotifClusterCollection
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
import networkx as nx
import cdt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from tqdm import tqdm
# import r2_score
from atac_rna_data_processing.io.causal_lib import preprocess_net
import matplotlib.pyplot as plt
import networkx.algorithms.community as nxcom

motif = NrMotifV1.load_from_pickle(
    pkg_resources.resource_filename("atac_rna_data_processing", "data/NrMotifV1.pkl")
)
motif_clusters = motif.cluster_names

# Load gencode_hg38 from feather file
gencode_hg38 = pd.read_feather("./gencode.v40.hg38.feather")
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
        """Summarize the jacobian of a gene."""
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

    def get_gene_by_motif(self):
        """Get the jacobian of all genes by motif."""
        if os.path.exists(os.path.join(self.interpret_cell_dir, f"{self.celltype}.zarr")):
            self._zarr_data = zarr.open(
                os.path.join(self.interpret_cell_dir, f"{self.celltype}.zarr"), mode="a"
            )
            if 'gene_by_motif' in self._zarr_data.keys():
                self.gene_by_motif = pd.DataFrame(
                    self._zarr_data["gene_by_motif"][:], columns=self.features)
            else:
                jacobs = []
                for g in tqdm(self.gene_annot.gene_name.unique()):
                    for j in self.get_gene_jacobian(g):
                        jacobs.append(j.motif_summary().T)
                jacobs_df = pd.concat(jacobs, axis=1).T
                # save to zarr
                self._zarr_data.array("gene_by_motif", jacobs_df.values, dtype="float32")
                self.gene_by_motif = jacobs_df
        elif os.path.exists(
            f"{self.interpret_dir}/{self.celltype}_gene_by_motif.feather"
        ):
            self.gene_by_motif = pd.read_feather(
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
            self.gene_by_motif = jacobs_df

        if isinstance(self.gene_by_motif, pd.DataFrame):
            self.gene_by_motif = GeneByMotif(self.celltype, self.interpret_dir, self.gene_by_motif)
            if os.path.exists(os.path.join(self.interpret_cell_dir, f"{self.celltype}.zarr")):
                if 'gene_by_motif_corr' in self._zarr_data.keys():
                    self.gene_by_motif.corr = pd.DataFrame(self._zarr_data["gene_by_motif_corr"][:], columns=self.features, index=self.features)
                    
                else:
                    # compute corr and save to zarr also
                    self._zarr_data.array("gene_by_motif_corr", self.gene_by_motif.get_corr().values, dtype="float32")

        return self.gene_by_motif

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
    
    def __repr__(self) -> str:
        return f"""Celltype: {self.celltype}
        Jacob shape: {self.data.shape}
        """

    def get_corr(self, method="spearman", diagal_to_zero=True):
        """Get the motif correlation."""
        if not hasattr(self, "corr"):
            corr = self.data.corr(method=method)
            if diagal_to_zero:
                corr = self.set_diagnal_to_zero(corr)
            self.corr = corr
            return self.corr
        else:
            return self.corr
            
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
