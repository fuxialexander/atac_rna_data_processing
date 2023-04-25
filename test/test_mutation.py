#%%
from atac_rna_data_processing.io.mutation import *
from atac_rna_data_processing.io.region import *
hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')
gbm_gwas = read_gwas_catalog(hg38, '/home/xf2217/Projects/geneformer_nova/analyze_gbm/gwas-association-downloaded_2023-04-25-EFO_0000519.tsv')
# %%
gbm_gwas = gbm_gwas.drop_duplicate_positions()
# %%
gbm_gwas
# %%
len(gbm_gwas.Ref_seq[0].sequences)
# %%
