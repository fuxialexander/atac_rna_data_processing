# %%
import sys
sys.path.append('..')
from atac_rna_data_processing.rna import RNA
# %%
a = RNA(sample='test', assembly='hg38', version=40, transform=True)
# %%
a
# %%
a.get_tss_atac_idx('chr1', 778769)
# %%
a.get_gene('RET')
# %%
