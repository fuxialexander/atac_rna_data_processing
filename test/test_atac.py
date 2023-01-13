# %%
import sys
sys.path.append('..')
# %%
from atac_rna_data_processing.atac import ATAC
# %%
atac = ATAC('test', 'hg38')
# %%
atac.export_data()
# %%
