# %%
import sys
sys.path.append('..')
from atac_rna_data_processing.rna import read_rna
# %%
a = read_rna("test.rna.csv", transform=True)
# %%
a
# %%
a.TPM.max()
# %%
