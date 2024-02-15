#%%
from scipy.sparse import load_npz
import numpy as np
import pandas as pd
import seaborn as sns

# %%
root = '/pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/'
data = load_npz(f'{root}/118.watac.npz').toarray()
# %%
exp = np.load(f'{root}/118.exp.npy')
#%%
exp_df = pd.read_feather(f'{root}/118.exp.feather')
#%%
atac = pd.read_csv(f'{root}/118.csv')
#%%
atac['exp_n'] = exp[:,1]
atac['exp_p'] = exp[:,0]
#%%
from pyranges import PyRanges as pr 

d = pr(atac).join(pr(exp_df.drop(['Start_b', 'End_b'], axis=1), 'chr', 'Start', 'End')).df
#%%
d.query('gene_name!="-1"')[['gene_name', 'TPM', 'exp_p', 'Strand']].query('Strand=="+"').query('TPM!=exp_p').drop_duplicates()
# %%
sns.scatterplot(x=data[:,282][exp[:,1]>0], y=exp[:,1][exp[:,1]>0], s=1)
# %%
np.corrcoef(data[:,282][exp[:,1]>0], exp[exp[:,1]>0,1])
# %%
from scipy.stats import spearmanr
spearmanr(data[:,282][exp[:,1]>0], exp[exp[:,1]>0,1])
# %%
