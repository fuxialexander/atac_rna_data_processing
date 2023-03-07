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
motif.get_motifcluster_list()
# %%
motif.get_motif_cluster_by_name(
    'PAX/1').motifs.get_motif('PAX6_MOUSE.H11MO.0.C').gene_name
#%%
import numpy as np
import pandas as pd
motif_cluster_gene_list = np.unique(np.concatenate(list(motif.cluster_gene_list.values())))
motif_cluster_gene_df = pd.Series(motif.cluster_gene_list).explode().reset_index().rename({'index':'cluster', 0:'gene_name'}, axis=1)
#%%

human_tf_collection = pd.read_csv('/home/xf2217/Projects/motif_databases/TF_Information.txt',sep='\t')
tf_list = pd.read_csv("../human/tf_list.csv")
human_tf_collection = pd.merge(tf_list, human_tf_collection, left_on='gene_name', right_on='TF_Name', how='left')
human_tf_collection['in_cluster'] = human_tf_collection['gene_name'].apply(lambda x: x in motif_cluster_gene_list)
human_tf_collection[['gene_name', 'in_cluster', 'Motif_ID']].query('in_cluster == False & ~Motif_ID.isnull() & Motif_ID!="."').to_csv('motif_non_cluster_tf_list.csv', index=False)
human_tf_collection[['gene_name', 'in_cluster', 'Motif_ID']].query('in_cluster == True & ~Motif_ID.isnull() & Motif_ID!="."').to_csv('motif_cluster_tf_list.csv', index=False)
human_tf_collection_not_in_cluster = human_tf_collection[['gene_name', 'in_cluster', 'Motif_ID']].query('in_cluster == False & ~Motif_ID.isnull() & Motif_ID!="."')
human_tf_collection_in_cluster_dict = human_tf_collection[['gene_name', 'in_cluster', 'Motif_ID']].query('in_cluster == True & ~Motif_ID.isnull() & Motif_ID!="."').set_index('Motif_ID').to_dict()['gene_name']
#%%
tomtom = pd.read_csv('/home/xf2217/Projects/motif_databases/tomtom.all.txt', sep='\t')
tomtom.rename({'#Query ID': 'Query_ID', 'Target ID': 'Target_ID'}, axis=1, inplace=True)
tomtom = tomtom[['Query_ID', 'Target_ID', 'q-value']].query('Query_ID.str.endswith("_2.00") & Target_ID.str.endswith("_2.00")')
tomtom = pd.concat([tomtom, tomtom.rename({'Query_ID': 'Target_ID', 'Target_ID': 'Query_ID'}, axis=1)]).drop_duplicates()
tomtom['in_cluster'] = tomtom['Target_ID'].apply(lambda x: human_tf_collection_in_cluster_dict.get(x, False))
#%%
def query_tomtom(query):
    df = tomtom.query('Query_ID == @query & in_cluster').sort_values('q-value')
    if df.shape[0]>0:
        return df.iloc[0].in_cluster
    else:
        return False
def query_cluster_of_gene(gene_name):
    return ','.join(motif_cluster_gene_df.query('gene_name == @gene_name').cluster.values)
human_tf_collection_not_in_cluster['in_cluster'] = human_tf_collection_not_in_cluster['Motif_ID'].apply(query_tomtom)
human_tf_collection_not_in_cluster['cluster'] = human_tf_collection_not_in_cluster.query('in_cluster!=False').in_cluster.apply(query_cluster_of_gene)
human_tf_collection_not_in_cluster.to_csv('motif_non_cluster_tf_list.csv', index=False)
human_tf_collection_in_cluster = human_tf_collection[['gene_name', 'in_cluster', 'Motif_ID']].query('in_cluster == True & ~Motif_ID.isnull() & Motif_ID!="."')
human_tf_collection_in_cluster['cluster'] = human_tf_collection_in_cluster['gene_name'].apply(query_cluster_of_gene)

#%%
pd.concat((human_tf_collection_in_cluster, human_tf_collection_not_in_cluster))[[
    'gene_name', 'in_cluster', 'cluster'
]].drop_duplicates().query('in_cluster!=False').to_csv('../human/tf_list.motif_cluster.csv', index=False)













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
