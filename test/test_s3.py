# %%
import sys
sys.path.append('..')
from atac_rna_data_processing.io.celltype import Celltype, GETCellType
#%%
import s3fs
import pandas as pd

from atac_rna_data_processing.config.load_config import load_config
from atac_rna_data_processing.io.celltype import GETCellType
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
GET_CONFIG = load_config(
    "../atac_rna_data_processing/config/GET"
)
GET_CONFIG.celltype.jacob = True
GET_CONFIG.celltype.num_cls = 2
GET_CONFIG.celltype.input = True
GET_CONFIG.celltype.embed = False
s3 = s3fs.S3FileSystem(anon=True)
GET_CONFIG.s3_file_sys = s3
s3_uri = "s3://2023-get-xf2217/get_demo"
GET_CONFIG.celltype.data_dir = (
    f"{s3_uri}/pretrain_human_bingren_shendure_apr2023/fetal_adult/"
)
GET_CONFIG.celltype.interpret_dir = (
    f"{s3_uri}/Interpretation_all_hg38_allembed_v4_natac/"
)
GET_CONFIG.motif_dir = f"{s3_uri}/interpret_natac/motif-clustering/"
cell_type_annot = pd.read_csv(
    GET_CONFIG.celltype.data_dir.split("fetal_adult")[0]
        + "data/cell_type_pretrain_human_bingren_shendure_apr2023.txt"
)
# %%
c_dict = {}
for i in range(23, 30):
    # check if i.csv in f"{s3_uri}/pretrain_human_bingren_shendure_apr2023/fetal_adult/"
    if s3.exists(f"{GET_CONFIG.celltype.data_dir}{i}.csv"):
        c_dict[i] = GETCellType(str(i), GET_CONFIG)

# %%
c_dict
# %%
for i in range(30,33):
    if s3.exists(f"{GET_CONFIG.celltype.data_dir}{i}.csv"[5:]):
        c_dict[i] = GETCellType(str(i), GET_CONFIG)
# %%
from tqdm import tqdm
for c in tqdm(c_dict):
    c_dict[c].get_gene_by_motif()
# %%
for c in tqdm(c_dict):
    c_dict[c].gene_by_motif.get_causal()
# %%
