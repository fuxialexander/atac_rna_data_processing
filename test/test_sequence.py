# %%
from atac_rna_data_processing.io.region import *
from atac_rna_data_processing.io.atac import read_bed4

hg38 = Genome('hg38', 'hg38.fa')
atac = read_bed4("test.atac.bed").as_df().head(100000)
#%%
atac = GenomicRegionCollection(hg38, atac)
#%%
seq = atac.center_expand(400).collect_sequence()
# %%
seq.save_npz("test_sequence")
# %%
from scipy.sparse import load_npz
load_npz("test_sequence_400.npz")

# %%
