# %%
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
import sys
sys.path.append('..')

motif = NrMotifV1("/home/xf2217/Projects/motif_databases/motif-clustering/")
# %%
motif.get_motif_list()
# %%
motif.get_motif('LHX6_homeodomain_3').plot_logo()

# %%
motif.get_motifcluster_list()
# %%
motif.get_motif_cluster_by_name(
    'PAX/1').motifs.get_motif('PAX9_PAX_1').plot_logo()
# %%
