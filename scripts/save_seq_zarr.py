# %%
import torch
from atac_rna_data_processing.io.region import *
import sys
#%%
sys.path.append('/manitou/pmg/users/xf2217/get_model')

hg38 = Genome('hg38', 'hg38.fa')
#%%
# for loop to get all hg38 sequence scanned, handle
chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7',
        'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14',
        'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21',
        'chr22', 'chrX', 'chrY']
# %%
for chr in chrs:
    print(chr)
    data = hg38.tiling_region(chr, 2000000, 2000000).collect_sequence(upstream=50, downstream=50)
    data.save_zarr_group(zarr_root='/pmglocal/xf2217/hg38_seq.zarr', key=chr, chunks=(100, 2000100, 4), target_length=2000100)
