#%%
import numpy as np 
# %%
from scipy.sparse import load_npz
data = load_npz('./test.natac.npz')
# %%
data.shape
# %%
data_subsample = data[np.random.choice(data.shape[0], 50000, replace=False), :]
# %%
# clustering of data_subsample on columns and sort
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

data_subsample_dist = pdist(data_subsample.todense().T, metric='euclidean')
data_subsample_link = linkage(data_subsample_dist, method='ward')
columns_sorted = dendrogram(data_subsample_link, no_plot=True)['leaves']
# %%
columns_sorted
# %%
def get_sample(data, columns_sorted, return_noise=False):
    if return_noise:
        noise = np.random.normal(0, 1, (200,283))
        return noise
    idx = np.random.random_integers(0, data.shape[0]-200)
    return data[idx:idx+200, np.array(columns_sorted)].todense().T
# %%
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(get_sample(data, columns_sorted))
# remove colorbar
cbar = ax.collections[0].colorbar
cbar.remove()

# %%
# generate 1000 samples and make it a gif
import imageio
from tqdm import tqdm
images = []
for i in tqdm(range(50)):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(get_sample(data, columns_sorted))
    # remove colorbar
    cbar = ax.collections[0].colorbar
    cbar.remove()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('tmp.png', dpi=100)
    images.append(imageio.imread('tmp.png'))
imageio.mimsave('heatmap.gif', images, duration=20)
# %%
# play gif
from IPython.display import Image
Image(filename='heatmap.gif')
# %%
from atac_rna_data_processing.io.sequence import DNASequence, DNASequenceCollection
# %%
seq_collection = DNASequenceCollection.from_fasta('test.fa')
# %%
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
motifs = NrMotifV1.load_from_pickle('/home/xf2217/Projects/atac_rna_data_processing/atac_rna_data_processing/data/NrMotifV1.pkl')
#%%
seq_collection.sequences[0].header='test'
# %%
seq_collection.scan_motif(motifs)
# %%
# %%
