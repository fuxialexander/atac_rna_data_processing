# %%
import sys
sys.path.append('..')
from atac_rna_data_processing.io.gencode import Gencode
#%%
gencode_hg38 = Gencode("hg38", 40)
# %%
gencode_hg38.gtf
# %%
gencode_hg38.get_gene("TERT")
# %%
gencode_hg38.get_gene_id("ENSG00000000003")
# %%
gencode_hg38.get_gene("TERT")
# %%
