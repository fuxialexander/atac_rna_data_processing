# %%
from sklearn.isotonic import spearmanr
from atac_rna_data_processing.io.celltype import GETCellType
from atac_rna_data_processing.config.load_config import load_config
GET_CONFIG = load_config('/home/xf2217/Projects/atac_rna_data_processing/atac_rna_data_processing/config/GET')
from tqdm import tqdm
# %%
cell = GETCellType('k562_peak_400', GET_CONFIG)
#%%
cell
#%%
cell.get_gene_idx('MYC')
# %%
cell.get_gene_tss_start('MYC')
# %%
cell.get_gene_annot('MYC')
# %%
cell.get_gene_jacobian('MYC')
# %%
myc0 = cell.get_gene_jacobian('MYC', False)[0]
myc1 = cell.get_gene_jacobian('MYC', True)[0]
# %%
import seaborn as sns
sns.scatterplot(x=myc0.motif_summary('signed_absmean'), y=myc1.motif_summary('signed_absmean'))

# %%
sns.scatterplot(x=myc0.motif_summary('mean'), y=myc1.motif_summary('mean'))

# %%
jacobs = []
for gene in tqdm(cell.gene_annot.gene_name.unique()):
    jacobs.append(cell.get_gene_jacobian(gene, True))

# %%
import numpy as np
jacobs = np.concatenate(jacobs).flatten()
#%%
jacobs[0].motif_summary('mean')
# %%
import pandas as pd
motif_df_mean = pd.DataFrame([j.motif_summary('mean') for j in tqdm(jacobs)])
motif_df_absmean = pd.DataFrame([j.motif_summary('absmean') for j in tqdm(jacobs)])
#%%
motif_df_tss = pd.DataFrame([j.data.iloc[100,4:] for j in tqdm(jacobs)])
# %%
(motif_df_mean.T/cell.preds).T.plot(x='MECP2', y='MBD2', kind='scatter',s=1)
#%%
(motif_df_mean.T/cell.preds).T.plot(x='ZFX', y='TFAP2/2', kind='scatter',s=1)
# %%
(motif_df_absmean.T/cell.preds).T.plot(x='HIF', y='GC-tract', kind='scatter',s=1)
# %%
sns.scatterplot(x=motif_df_mean.mean(1), y=cell.preds,s=1)
# add x=0 line
sns.lineplot(x=[0,0], y=[0,4], color='red')
# %%
sns.scatterplot(x=motif_df_absmean.mean(1), y=cell.preds,s=1)
sns.lineplot(x=[0,0], y=[0,4], color='red')
# %%
sns.scatterplot(x=motif_df_absmean.iloc[:,0], y=cell.preds,s=1)
# %%
for i in motif_df_absmean.items():
    if np.corrcoef(i[1], motif_df_absmean['Accessibility'])[0,1]>0.85:
        print(i[0], np.corrcoef(i[1], motif_df_absmean['Accessibility'])[0,1])
# %%
sns.scatterplot(x=motif_df_mean['ETS/2'], y=motif_df_mean['ZFX'], hue=cell.preds,s=2)
# %%
from scipy.stats import spearmanr
spearmanr(motif_df_mean['TFAP2/2'], motif_df_mean['ZFX'])
# %%

# %%
sns.scatterplot(x=motif_df_mean.values.reshape(-1)*200-motif_df_tss.values.reshape(-1), y=motif_df_tss.values.reshape(-1),s=1, alpha=0.1)
# %%
sns.scatterplot(x=motif_df_absmean.values.reshape(-1), y=motif_df_tss.values.reshape(-1),s=1)
# %%
# %%
sns.scatterplot(x=motif_df_absmean.values.reshape(-1), y=motif_df_tss.abs().values.reshape(-1),s=1)
# %%
spearmanr(motif_df_tss['TFAP2/2'], motif_df_tss['ZFX'])
# %%
for i in motif_df_absmean.items():
    if spearmanr(i[1], motif_df_absmean['Accessibility'])[0]>0.75:
        print(i[0], spearmanr(i[1], motif_df_absmean['Accessibility'])[0])
# %%
def set_diagnal_to_zero(df):
    for i in range(df.shape[0]):
        df.iloc[i,i] = 0
    return df

def plot_corr(df, method='spearman', topk=20):
    corr = df.corr(method=method)
    corr = set_diagnal_to_zero(corr)
    corr_p = corr.iloc[corr.abs().sum(0).argsort()[-topk:],corr.abs().sum(0).argsort()[-topk:]]
    sns.clustermap(corr_p,  figsize=(20,20), method='ward')
    return corr
corr = plot_corr(motif_df_mean, topk=30)
# %%
corr['CTCF'].sort_values(ascending=False).head(20)
# %%
corr['GATA'].sort_values(ascending=False).head(20)

# %%
# tsne of motif_df_mean
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(motif_df_absmean)
# %%
sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=cell.preds,s=2)
# %%