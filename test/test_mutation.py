#%%
from atac_rna_data_processing.io.mutation import *
from atac_rna_data_processing.io.region import *
hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
motif = NrMotifV1("/home/xf2217/Projects/motif_databases/motif-clustering/")
# %%
gbm_gwas = read_gwas_catalog(hg38, '/home/xf2217/Projects/geneformer_nova/analyze_gbm/gwas-association-downloaded_2023-04-25-EFO_0000519.tsv')

# %%
gbm_gwas.get_motif_diff(motif)
# %%
alt_motif, ref_motif = gbm_gwas.get_motif_diff(motif).values()
# %%
alt_motif.values-ref_motif.values
# %%
