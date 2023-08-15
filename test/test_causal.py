#%%
import sys
sys.path.append('/manitou/pmg/users/xf2217/atac_rna_data_processing/')
import os
import re

import numpy as np
import pandas as pd
import scanpy as sc
import zarr 
import seaborn as sns
import matplotlib.pyplot as plt
from atac_rna_data_processing.config.load_config import load_config
from atac_rna_data_processing.io.celltype import GETCellType
from pyranges import PyRanges as pr

#plt.style.use('manuscript.mplstyle')
#%%
GET_CONFIG = load_config('/manitou/pmg/users/xf2217/atac_rna_data_processing/atac_rna_data_processing/config/GET')
GET_CONFIG.celltype.jacob=True
GET_CONFIG.celltype.num_cls=2
GET_CONFIG.celltype.input=True
GET_CONFIG.celltype.embed=True
GET_CONFIG.celltype.data_dir = '/pmglocal/xf2217/tmp/fetal_adult/'
GET_CONFIG.celltype.interpret_dir='/pmglocal/xf2217/tmp/Interpretation_all_hg38_allembed_v4_natac'
#%%
celltype = sys.argv[1]
cell = GETCellType(celltype, GET_CONFIG)
cell.get_gene_by_motif()
print(cell.gene_by_motif.data.shape)

causal = cell.gene_by_motif.get_causal(permute_columns=True, n=3, overwrite=True)


# %%
