import os
import argparse

import numpy as np
import pandas as pd
import pkg_resources
import sys
sys.path.append('../')
import zarr
from scipy.sparse import csr_matrix, load_npz, coo_matrix
from tqdm import tqdm
import zarr
from atac_rna_data_processing.config.load_config import *
from atac_rna_data_processing.io.gene import TSS, Gene, GeneExp
from atac_rna_data_processing.io.motif import MotifClusterCollection
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1

# Load gencode_hg38 from feather file
gencode_hg38 = pd.read_feather("../test/gencode.v40.hg38.feather")
gencode_hg38["Strand"] = gencode_hg38["Strand"].apply(lambda x: 0 if x == "+" else 1)
gene2strand = gencode_hg38.set_index("gene_name").Strand.to_dict()

parser = argparse.ArgumentParser(description='Process cell type name and root path.')
parser.add_argument('--cell_type_name', type=str, help='Name of the cell type')
parser.add_argument('--root_path', type=str, help='Root path')

args = parser.parse_args()

def load_gene_annot(cell_type_name, root_path):
    """Load gene annotations from feather file."""
    gene_feather_path = f"{root_path}/{cell_type_name}/allgenes/{cell_type_name}.exp.feather"
    if not os.path.exists(gene_feather_path):
        # construct gene annotation from gencode
        from pyranges import PyRanges as pr

        atac = pr(self.peak_annot, int64=True)
        # join the ATAC-seq data with the RNA-seq data
        exp = atac.join(
            pr(gencode_hg38, int64=True).extend(300), how="left"
        ).as_df()
        # save the data to feather file
        exp.reset_index(drop=True).to_feather(gene_feather_path)
    return
load_gene_annot(args.cell_type_name, args.root_path)
