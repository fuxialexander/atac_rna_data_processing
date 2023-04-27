#%%
from glob import glob
for f in glob("*/*/*.atac.motif.output.feather"):
    print(f, pd.read_feather(f)['AHR'].max())