# %%
import sys
sys.path.append('..')
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1

motif = NrMotifV1("/home/xf2217/Projects/motif_databases/motif-clustering/")
# %%
motif.get_motif_list()
# %%
motif.get_motif('LHX6_homeodomain_3').plot_logo()

# %%
len(motif.get_motifcluster_list())
# %%
motif.get_motif_cluster_by_name(
    'PAX/1').motifs.get_motif('PAX6_MOUSE.H11MO.0.C').gene_name
# %%
from MOODS.tools import bg_from_sequence_dna
from atac_rna_data_processing.io.region import Genome 
hg19 = Genome('hg19', '/home/xf2217/Projects/common/hg19.fasta')
bg_from_sequence_dna(hg19.get_sequence(2, 1, 1000000000).seq, 0.0001)
# %%
hg19.genome_seq.keys()
# %%
motif.scanner
# %%
