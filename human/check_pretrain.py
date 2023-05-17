#%%
import pandas as pd
from scipy.sparse import load_npz
import numpy as np
from scipy.stats import pearsonr, spearmanr
from glob import glob
from pyranges import PyRanges as pr
# %%
# use glob to examine all files
for f in glob('*/output/*.atac.motif.output.feather'):
    b = pd.read_feather(f)
    data_type = f.split('/')[0]
    sample_name = f.split('/')[-1].split('.')[0]
    a = load_npz(f'{data_type}/output/{sample_name}.natac.npz')
    corr = pearsonr(x=a[2].toarray().flatten()[:-1], y=(b.iloc[2, 4:]/b.iloc[:,4:].max(0).values[0:282]))[0]
    print(f, corr)
#%%
# use glob to examine all fetal_adult files
for f in glob('*/bed/*.atac.motif.output.feather'):
    b = pd.read_feather(f)
    data_type = f.split('/')[0]
    sample_name = f.split('/')[-1].split('.')[0]
    a = load_npz(f'{data_type}/output/{sample_name}.natac.npz')
    corr = pearsonr(x=a[2].toarray().flatten()[:-1], y=(b.iloc[2, 4:]/b.iloc[:,4:].max(0).values[0:282]))[0]
    print(f, corr)

# %%

for f in glob('*/bed/*.atac.motif.output.feather')[0:5]:
    for f1 in glob('*/bed/*.atac.motif.output.feather')[0:5]:
        b = pd.read_feather(f)
        b1 = pd.read_feather(f1)
        data_type = f.split('/')[0]
        sample_name = f.split('/')[-1].split('.')[0]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.iloc[:,5:287].values.reshape(-1), y=x.df.iloc[:,292:292+282].values.reshape(-1))
        print(f, f1, corr)
# %%

for f in glob('k562/output/*.atac.motif.output.feather')[0:5]:
    for f1 in glob('TCGA/output/*.atac.motif.output.feather')[0:5]:
        b = pd.read_feather(f)
        b1 = pd.read_feather(f1)
        data_type = f.split('/')[0]
        sample_name = f.split('/')[-1].split('.')[0]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.iloc[:,5:287].values.reshape(-1), y=x.df.iloc[:,291:291+282].values.reshape(-1))
        print(f, f1, corr)
# %%

for f in glob('k562/output/*.atac.motif.output.feather')[0:5]:
    for f1 in glob('fetal_adult/bed/*.atac.motif.output.feather')[0:5]:
        b = pd.read_feather(f)
        b1 = pd.read_feather(f1)
        data_type = f.split('/')[0]
        sample_name = f.split('/')[-1].split('.')[0]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.iloc[:,5].values.reshape(-1), y=x.df.iloc[:,292].values.reshape(-1))
        print(f, f1, corr)
# %%

for f in glob('k562/output/*.atac.motif.output.feather')[0:5]:
    for f1 in glob('pbmc/output/*.atac.motif.output.feather')[0:5]:
        b = pd.read_feather(f)
        b1 = pd.read_feather(f1)
        data_type = f.split('/')[0]
        sample_name = f.split('/')[-1].split('.')[0]
        x = pr(b1.reset_index().head(10000), int64=True).join(pr(b.reset_index().head(10000), int64=True))
        corr = pearsonr(x=x.df.iloc[:,5:287].values.reshape(-1), y=x.df.iloc[:,291:291+282].values.reshape(-1))
        print(f, f1, corr)
# %%
for f in glob('k562/output/*.atac.motif.output.feather')[0:5]:
    for f1 in glob('GBM/output/*.atac.motif.output.feather')[0:5]:
        b = pd.read_feather(f)
        b1 = pd.read_feather(f1)
        data_type = f.split('/')[0]
        sample_name = f.split('/')[-1].split('.')[0]
        x = pr(b1.reset_index().head(10000), int64=True).join(pr(b.reset_index().head(10000), int64=True))
        corr = pearsonr(x=x.df.iloc[:,5:287].values.reshape(-1), y=x.df.iloc[:,291:291+282].values.reshape(-1))
        print(f, f1, corr)
# %%
for f in glob('TCGA/output/*.atac.motif.output.feather')[0:5]:
    for f1 in glob('pbmc/output/*.atac.motif.output.feather')[0:5]:
        b = pd.read_feather(f)
        b1 = pd.read_feather(f1)
        data_type = f.split('/')[0]
        sample_name = f.split('/')[-1].split('.')[0]
        x = pr(b1.reset_index().head(10000), int64=True).join(pr(b.reset_index().head(10000), int64=True))
        corr = pearsonr(x=x.df.iloc[:,5:287].values.reshape(-1), y=x.df.iloc[:,291:291+282].values.reshape(-1))
        print(f, f1, corr)
# %%
# %%
for f in glob('TCGA/output/*.atac.motif.output.feather')[0:5]:
    for f1 in glob('TFAtlas/output/*.atac.motif.output.feather')[0:5]:
        b = pd.read_feather(f)
        b1 = pd.read_feather(f1)
        data_type = f.split('/')[0]
        sample_name = f.split('/')[-1].split('.')[0]
        x = pr(b1.reset_index().head(10000), int64=True).join(pr(b.reset_index().head(10000), int64=True))
        corr = pearsonr(x=x.df.iloc[:,5:287].values.reshape(-1), y=x.df.iloc[:,291:291+282].values.reshape(-1))
        print(f, f1, corr)
# %%
for f in glob('TCGA/output/*.csv'):
    for f1 in glob('TFAtlas/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        br = np.load(f'{data_type1}/output/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/output/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.pos.values.reshape(-1), y=x.df.pos_b.values.reshape(-1))[0]
        corrneg = pearsonr(x=x.df.neg.values.reshape(-1), y=x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)


# %%
# %%
for f in glob('GBM/output/*.csv'):
    for f1 in glob('TFAtlas/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        br = np.load(f'{data_type1}/output/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/output/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.pos.values.reshape(-1), y=x.df.pos_b.values.reshape(-1))[0]
        corrneg = pearsonr(x=x.df.neg.values.reshape(-1), y=x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)
# %%
for f in glob('TCGA/output/*.csv'):
    for f1 in glob('k562/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        br = np.load(f'{data_type1}/output/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/output/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.pos.values.reshape(-1), y=x.df.pos_b.values.reshape(-1))[0]
        corrneg = pearsonr(x=x.df.neg.values.reshape(-1), y=x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)

# %%
# %%
for f in glob('TCGA/output/*.csv'):
    for f1 in glob('k562/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        br = np.load(f'{data_type1}/output/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/output/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.pos.values.reshape(-1), y=x.df.pos_b.values.reshape(-1))[0]
        corrneg = pearsonr(x=x.df.neg.values.reshape(-1), y=x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)
# %%
# %%
for f in glob('fetal_adult/bed/*.csv'):
    for f1 in glob('k562/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        if not os.path.exists(f'fetal_adult/bed/{s}.exp.npy'):
            print(s)
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        s1 = f1.split('/')[-1].split('.')[0]
        br = np.load(f'{data_type1}/bed/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/output/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.pos.values.reshape(-1), y=x.df.pos_b.values.reshape(-1))[0]
        corrneg = pearsonr(x=x.df.neg.values.reshape(-1), y=x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)
# %%
# %%
for f in glob('fetal_adult/bed/*.csv'):
    for f1 in glob('TCGA/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        if not os.path.exists(f'{data_type1}/bed/{s}.exp.npy'):
            continue
        if not os.path.exists(f'{data_type2}/output/{s1}.exp.npy'):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        br = np.load(f'{data_type1}/bed/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/output/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.pos.values.reshape(-1), y=x.df.pos_b.values.reshape(-1))[0]
        corrneg = pearsonr(x=x.df.neg.values.reshape(-1), y=x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)
# %%
# %%
for f in glob('fetal_adult/bed/*.csv'):
    for f1 in glob('fetal_adult/bed/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        if not os.path.exists(f'fetal_adult/bed/{s}.exp.npy'):
            continue
        if not os.path.exists(f'fetal_adult/bed/{s1}.exp.npy'):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        
        br = np.load(f'{data_type1}/bed/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/bed/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.pos.values.reshape(-1), y=x.df.pos_b.values.reshape(-1))[0]
        corrneg = pearsonr(x=x.df.neg.values.reshape(-1), y=x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)
# %%
# %%
for f in glob('fetal_adult/bed/*.csv'):
    for f1 in glob('lymph/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        if not os.path.exists(f'fetal_adult/bed/{s}.exp.npy'):
            continue
        if not os.path.exists(f'lymph/output/{s1}.exp.npy'):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        br = np.load(f'{data_type1}/bed/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/output/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = spearmanr(x.df.pos.values.reshape(-1), x.df.pos_b.values.reshape(-1))[0]
        corrneg = spearmanr(x.df.neg.values.reshape(-1), x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)
# %%
# %%
for f in glob('pbmc/output/*.csv'):
    for f1 in glob('lymph/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        if not os.path.exists(f'pbmc/output/{s}.exp.npy'):
            continue
        if not os.path.exists(f'lymph/output/{s1}.exp.npy'):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        br = np.load(f'{data_type1}/output/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/output/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.pos.values.reshape(-1), y=x.df.pos_b.values.reshape(-1))[0]
        corrneg = pearsonr(x=x.df.neg.values.reshape(-1), y=x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)
# %%
# %%
for f in glob('k562/output/*.csv'):
    for f1 in glob('k562/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        if not os.path.exists(f'{data_type1}/output/{s}.exp.npy'):
            continue
        if not os.path.exists(f'{data_type2}/output/{s1}.exp.npy'):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        br = np.load(f'{data_type1}/output/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/output/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = spearmanr(x.df.pos.values.reshape(-1), x.df.pos_b.values.reshape(-1))[0]
        corrneg = spearmanr(x.df.neg.values.reshape(-1), x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)
# %%
# %%
for f in glob('fetal_adult/bed/*.tf_atac.npy'):
    for f1 in glob('k562/output/*.tf_atac.npy'):
        if ('rna' in f) or ('rna' in f1):
            continue
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        if not os.path.exists(f'fetal_adult/bed/{s}.tf_exp.npy'):
            continue
        if not os.path.exists(f'k562/output/{s1}.tf_exp.npy'):
            continue
        b1 = np.load(f'{data_type1}/bed/{s}.tf_exp.npy')
        b2 = np.load(f'{data_type2}/output/{s1}.tf_exp.npy')
        a1 = np.load(f'{data_type1}/bed/{s}.tf_atac.npy')
        a2 = np.load(f'{data_type2}/output/{s1}.tf_atac.npy')
        corra = pearsonr(a1.reshape(-1), a2.reshape(-1))[0]
        corrb = pearsonr(b1.reshape(-1), b2.reshape(-1))[0]
        print(s, s1, corra, corrb)
# %%
for f in glob('k562/output/*.tf_atac.npy'):
    for f1 in glob('k562/output/*.tf_atac.npy'):
        if ('rna' in f) or ('rna' in f1):
            continue
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        if not os.path.exists(f'k562/output/{s}.tf_exp.npy'):
            continue
        if not os.path.exists(f'k562/output/{s1}.tf_exp.npy'):
            continue
        b1 = np.load(f'{data_type1}/output/{s}.tf_exp.npy')
        b2 = np.load(f'{data_type2}/output/{s1}.tf_exp.npy')
        a1 = np.load(f'{data_type1}/output/{s}.tf_atac.npy')
        a2 = np.load(f'{data_type2}/output/{s1}.tf_atac.npy')
        corra = pearsonr(a1.reshape(-1), a2.reshape(-1))[0]
        corrb = pearsonr(b1.reshape(-1), b2.reshape(-1))[0]
        print(s, s1, corra, corrb)
# %%
for f in glob('pretrain_human_bingren_shendure_apr2023/TCGA/*.csv'):
    for f1 in glob('pretrain_human_bingren_shendure_apr2023/GBM/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        data_type1 = f.split('/')[1]
        data_type2 = f1.split('/')[1]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]
        if not os.path.exists(f'pretrain_human_bingren_shendure_apr2023/TCGA/{s}.exp.npy'):
            continue
        if not os.path.exists(f'pretrain_human_bingren_shendure_apr2023/GBM/{s1}.exp.npy'):
            continue

        br = np.load(f'{data_type1}/output/{s}.exp.npy')
        br1 = np.load(f'{data_type2}/output/{s1}.exp.npy')
        b['pos'] = br[:,0]
        b1['pos'] = br1[:,0]
        b['neg'] = br[:,1]
        b1['neg'] = br1[:,1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr = pearsonr(x=x.df.pos.values.reshape(-1), y=x.df.pos_b.values.reshape(-1))[0]
        corrneg = pearsonr(x=x.df.neg.values.reshape(-1), y=x.df.neg_b.values.reshape(-1))[0]
        print(s, s1, corr, corrneg)
# %%
for f in glob('TCGA/output/*.csv'):
    for f1 in glob('TCGA/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]

        br = load_npz(f'{data_type1}/output/{s}.watac.npz').toarray()
        br1 = load_npz(f'{data_type2}/output/{s1}.watac.npz').toarray()
        b['m1'] = br[:,0]
        b['m2'] = br[:,1]
        b1['m1'] = br1[:,0]
        b1['m2'] = br1[:,1]
        b['atac'] = br[:,-1]
        b1['atac'] = br1[:,-1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr1 = pearsonr(x=x.df.m1.values.reshape(-1), y=x.df.m1_b.values.reshape(-1))[0]
        corr2 = pearsonr(x=x.df.m2.values.reshape(-1), y=x.df.m2_b.values.reshape(-1))[0]
        corratac = pearsonr(x=x.df.atac.values.reshape(-1), y=x.df.atac_b.values.reshape(-1))[0]
        print(s, s1, corr1, corr2, corratac)
# %%
# %%
for f in glob('k562/output/*count*.csv'):
    for f1 in glob('k562/output/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]

        br = load_npz(f'{data_type1}/output/{s}.watac.npz').toarray()
        br1 = load_npz(f'{data_type2}/output/{s1}.watac.npz').toarray()
        b['m1'] = br[:,0]
        b['m2'] = br[:,1]
        b1['m1'] = br1[:,0]
        b1['m2'] = br1[:,1]
        b['atac'] = br[:,-1]
        b1['atac'] = br1[:,-1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr1 = pearsonr(x=x.df.m1.values.reshape(-1), y=x.df.m1_b.values.reshape(-1))[0]
        corr2 = pearsonr(x=x.df.m2.values.reshape(-1), y=x.df.m2_b.values.reshape(-1))[0]
        corratac = pearsonr(x=x.df.atac.values.reshape(-1), y=x.df.atac_b.values.reshape(-1))[0]
        print(s, s1, corr1, corr2, corratac)
# %%
# %%
for f in glob('fetal_adult/bed/*.csv'):
    for f1 in glob('fetal_adult/bed/*.csv'):
        if ('rna' in f) or ('rna' in f1):
            continue
        b = pd.read_csv(f)
        b1 = pd.read_csv(f1)
        data_type1 = f.split('/')[0]
        data_type2 = f1.split('/')[0]
        s = f.split('/')[-1].split('.')[0]
        s1 = f1.split('/')[-1].split('.')[0]

        br = load_npz(f'{data_type1}/bed/{s}.watac.npz').toarray()
        br1 = load_npz(f'{data_type2}/bed/{s1}.watac.npz').toarray()
        b['m1'] = br[:,0]
        b['m2'] = br[:,1]
        b1['m1'] = br1[:,0]
        b1['m2'] = br1[:,1]
        b['atac'] = br[:,-1]
        b1['atac'] = br1[:,-1]
        x = pr(b1.reset_index(), int64=True).join(pr(b.reset_index(), int64=True))
        corr1 = pearsonr(x=x.df.m1.values.reshape(-1), y=x.df.m1_b.values.reshape(-1))[0]
        corr2 = pearsonr(x=x.df.m2.values.reshape(-1), y=x.df.m2_b.values.reshape(-1))[0]
        corratac = pearsonr(x=x.df.atac.values.reshape(-1), y=x.df.atac_b.values.reshape(-1))[0]
        print(s, s1, corr1, corr2, corratac)
# %%
