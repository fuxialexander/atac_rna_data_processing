# %%
import sys
sys.path.append('..')
from atac_rna_data_processing.region import *
#%%

hg19 = Genome('hg19', '/home/xf2217/Projects/common/hg19.fasta')
hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')

gr = GenomicRegion(hg19, 'chr1', 10000, 12000)
# %%
gr.sequence
# %%
hg19.chr_suffix
# %%
hg38.chr_suffix
# %%
pr = GenomicRegionCollection(genome=hg19, chromosomes = ['chr1', 'chr2'], starts = [10000, 20000], ends = [12000, 22000])
# %%
pr.as_df()
# %%
pr.collect_sequence(100,100)
# %%
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1

motifs = NrMotifV1("/home/xf2217/Projects/motif_databases/motif-clustering/")
#%%
scanner = prepare_scanner()
pr.scan_motif(motifs, )
# %%
