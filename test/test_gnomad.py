#%%
#This script is to download and load gnomAD data:
import sys
#The sys.path.append() method is used specifically to add a Path to the existing ones.
sys.path.append('..')
# %%To import the object ATAC from the script atac.py
from atac_rna_data_processing.io.mutation import tabix_query


# %%
# Example usage
tabix_query("https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.0/vcf/genomes/gnomad.genomes.v4.0.sites.chr17.vcf.bgz", "chr17", 7161779, 8187538, "output_file.vcf")


# %%
