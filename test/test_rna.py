# %%
import sys

from matplotlib import pyplot as plt
sys.path.append('..')
from atac_rna_data_processing.io.rna import RNA
#%%
import pandas as pd
k562_encode = pd.read_csv('k562_encode_rsem.tsv', sep='\t')
k562_encode['gene_id'] =  k562_encode.gene_id.str.split('.').str[0]
# %%
a = RNA(sample='../human/k562_overlap/k562_overlap', assembly='hg38', version=40, transform=False, tf_list="../human/tf_list.txt", id_or_name='gene_id', cnv_file='../human/k562/k562_cnv.bed')
a2 = RNA(sample='../human/k562/k562_dnase', assembly='hg38', version=40, transform=False, tf_list="../human/tf_list.txt", id_or_name='gene_id', cnv_file='../human/k562/k562_cnv.bed')
#%%
import numpy as np
b = np.load("/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/118.tf_exp.npy") # astrocyte
c = np.load("/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/GBM/SF9798.tf_exp.npy") # GBM
d = np.load("/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/k562_cut/k562_cut0.07.tf_atac.npy") # k562
#%%
ery = pd.read_feather("/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/155.exp.feather")
#%%
import pandas as pd
fetal = pd.read_csv("../../NEAT/fetal_promoter_exp.txt", sep=',').iloc[:,6:].drop('RowID', axis=1).set_index('ensembl_id')
fetal = np.log10(fetal+1)
#%%
merged = ery.merge(a.rna, left_on='gene_id', right_on='gene_id', how='inner')
merged.plot(kind='scatter', x='TPM_x', y='TPM_y', alpha=1, s=1)
# correlation
np.corrcoef(merged.TPM_x, merged.TPM_y)[0, 1]
#%%
import seaborn as sns
sns.scatterplot(x=b, y=c)
#%%
merged = a.rna.merge(k562_encode, left_on='gene_id', right_on='gene_id', how='inner')
merged['posterior_mean_count'] = np.log10(merged['posterior_mean_count']/merged['posterior_mean_count'].sum()*10e6+1)
merged['expected_count'] = np.log10(merged['expected_count']/merged['expected_count'].sum()*10e6+1)
merged['TPM_y'] = np.log10(merged['TPM_y']/merged['TPM_y'].sum()*10e6+1)
merged['FPKM'] = np.log10(merged['FPKM']/merged['TPM_y'].sum()*10e6+1)
#%%
#r2_score
from sklearn.metrics import r2_score
merged.plot(kind='scatter', x='TPM_x', y='FPKM', alpha=1, s=1)
r2_score(merged.TPM_x, merged.posterior_mean_count)
#%%
# calcuate correlation
np.corrcoef(merged.TPM_x, merged.FPKM)[0, 1]
#%%
import seaborn as sns
g = sns.scatterplot(x=b, y=a.tf_exp)
# g.set(xlabel='Astrocyte TF Exp', ylabel='K562 TF Exp')
# add correlation as title
g.set_title("Correlation: {}".format(np.corrcoef(b, a.tf_exp)[0, 1]))
#%%
g = sns.scatterplot(x=b, y=c)
g.set(xlabel='Astrocyte TF Exp', ylabel='GBM TF Exp')
#%%
g = sns.scatterplot(x=b, y=a.tf_exp)
g.set(xlabel='Astrocyte TF Exp', ylabel='K562 Bulk CNV TF Exp')
# x y axis name
#%%
g = sns.scatterplot(x=b, y=a.tf_exp)
g.set(xlabel='Astrocyte TF Exp', ylabel='K562 Bulk w/o CNV TF Exp')
# x y axis name
#%%
import pandas as pd
cnv = pd.read_csv('../human/k562/k562_cnv.bed', sep='\t', header=None)
cnv.columns = ['Chromosome', 'Start', 'End', 'fc']
cnv['fc'] = cnv['fc'].apply(lambda x: 2**x)
#%%
# for every gene, adjust the tpm by cnv

# %%
a.get_tss_atac_idx('chr1', 778769)
# %%
a.get_gene('RET')
# %%
a.rna
# %%
from pyranges import read_bed
atac = read_bed("test.atac.bed")
# %%
atac
# %%
from pyranges import PyRanges as pr
b  = atac.join(pr(a.rna), how = 'outer')
# %%
import seaborn as sns

covered_genes = b[b.Start!=-1].as_df().gene_name.unique()
uncovered_genes = b[b.Start==-1].as_df().groupby('gene_name').TPM.mean()
uncovered_genes = uncovered_genes[~uncovered_genes.index.isin(covered_genes)]
sns.displot(uncovered_genes)
# %%

