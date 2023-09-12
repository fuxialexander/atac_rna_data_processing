#%%
import sys
sys.path.append('/manitou/pmg/users/xf2217/atac_rna_data_processing/')

from atac_rna_data_processing.config.load_config import load_config
from atac_rna_data_processing.io.celltype import GETCellType
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
import numpy as np
#%%
motif = NrMotifV1('/manitou/pmg/users/xf2217/interpret_natac/motif-clustering')

#%%
GET_CONFIG = load_config('/manitou/pmg/users/xf2217/atac_rna_data_processing/atac_rna_data_processing/config/GET')
GET_CONFIG.celltype.jacob=True
GET_CONFIG.celltype.num_cls=2
GET_CONFIG.celltype.input=True
GET_CONFIG.celltype.embed=True
GET_CONFIG.celltype.data_dir = '/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/'
GET_CONFIG.celltype.interpret_dir='/manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac/'
# %%
celltype = '118'
cell = GETCellType(celltype, GET_CONFIG)

# %%
cell.plot_gene_regions('MYC', plotly=True)
# %%
cell.plot_gene_motifs('MYC', motif, overwrite=False)

# %%
# plot dash_bio.Clustergram on cell.gene_by_motif.corr
from dash_bio import Clustergram
Clustergram(
    data=cell.gene_by_motif.corr,
    column_labels=list(cell.gene_by_motif.corr.columns.values),
    row_labels=list(cell.gene_by_motif.corr.index),
    height=800,
    width=1000,
    hidden_labels=['row', 'col'],
    link_method='ward',
    display_ratio=0.1,
    color_map='rdbu_r',
)

#%%
cell.plot_motif_subnet(motif, 'SIX/1', type='neighbors', threshold=0.1)
# %%
