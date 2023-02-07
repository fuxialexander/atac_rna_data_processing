# %%
import sys
sys.path.append('..')
from atac_rna_data_processing.region import *
#%%

hg19 = Genome('hg19', '/home/xf2217/Projects/common/hg19.fasta')
hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')

gr = GenomicRegion(hg19, 'chr1', 350000, 352000)
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

# %%
from pyranges import read_bed
atac = read_bed("test.atac.bed").as_df()

# %%
# local parallel using ipyparallel
import ipyparallel as ipp
with ipp.Cluster(n=14) as context:
    rc = context[:]
    rc.block = True
    rc.execute("import sys; sys.path.append('..')")
    rc.execute("from atac_rna_data_processing.region import *")
    rc.execute("from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1")
    rc.execute("motifs = NrMotifV1('/home/xf2217/Projects/motif_databases/motif-clustering/')")
    rc.execute("hg19 = Genome('hg19', '/home/xf2217/Projects/common/hg19.fasta')")
    rc.execute("hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')")

    rc.scatter('atac', atac)
    rc.execute("atac = GenomicRegionCollection(hg38, atac)")
    rc.execute("result = atac.scan_motif(motifs)")
    results = pd.concat(rc.gather('result'))
# %%
atac_seq = [s.padding(target_length=1000, left=50).one_hot for s in GenomicRegionCollection(hg38, atac).collect_sequence()]
# %%

# %%
atac_seq
# %%
