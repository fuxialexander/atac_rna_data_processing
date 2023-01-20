#This script is to compute a peak-by-motif binding score matrix:

# %%The python sys module provides functions and variables which are used to manipulate different parts of the Python Runtime Environment.
# It lets us access system-specific parameters and functions.
import sys
#The sys.path.append() method is used specifically to add a Path to the existing ones.
sys.path.append('..')
# %%To import the object ATAC from the script atac.py
from atac_rna_data_processing.atac import ATAC
# %%
atac = ATAC('test', 'hg38')
# %%Exports the data to a YAML file, a csv file and a npz file,
atac.export_data()
# %%
