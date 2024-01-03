#This script is to compute a peak-by-motif binding score matrix:

# %%The python sys module provides functions and variables which are used to manipulate different parts of the Python Runtime Environment.
# It lets us access system-specific parameters and functions.
import sys
#The sys.path.append() method is used specifically to add a Path to the existing ones.
sys.path.append('..')
# %%To import the object ATAC from the script atac.py
from atac_rna_data_processing.io.atac import ATAC
# %%
from atac_rna_data_processing.io.gencode import Gencode
atac = ATAC('../human/k562_overlap/k562_overlap', 'hg38', tf_list="../human/tf_list.txt", scanned_motif=True)
# %%Exports the data to a YAML file, a csv file and a npz file,
atac.export_data()

# %%
atac.export_data_to_zarr()


# %%
from scipy.sparse import load_npz
load_npz('../human/k562/k562_dnase.natac.npz')
# %%
atac.motif_data
# %%
