# this file generate training data for seq2motif model
# specifically:
# 1. Given a list of regions and a genome assembly, generate a list of sequences (one-hot encoded numpy array) as inputs
# 2. Given a list of sequences, generate a list of motif scores as targets
# %%
import sys
sys.path.append('..')
import ipyparallel as ipp
import pandas as pd
from pyranges import PyRanges as pr
from pyranges import read_bed
from scipy.sparse import load_npz, save_npz, vstack
from tqdm import tqdm
from atac_rna_data_processing.atac import *
from atac_rna_data_processing.gene import Gene
from atac_rna_data_processing.io.gencode import Gencode
from atac_rna_data_processing.io.nr_motif_v1 import *
from atac_rna_data_processing.motif import *
from atac_rna_data_processing.region import *



# %%
motifs = NrMotifV1("/home/xf2217/Projects/motif_databases/motif-clustering/")
hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')

# %%
# local parallel using ipyparallel
def parallel_scan_NrMotifV1(df):
    with ipp.Cluster(n=14) as context:
        rc = context[:]
        rc.block = True
        rc.execute("import sys; sys.path.append('..')")
        rc.execute("from atac_rna_data_processing.region import *")
        rc.execute(
            "from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1")
        rc.execute(
            "motifs = NrMotifV1('/home/xf2217/Projects/motif_databases/motif-clustering/')")
        rc.execute(
            "hg19 = Genome('hg19', '/home/xf2217/Projects/common/hg19.fasta')")
        rc.execute(
            "hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')")
        rc.scatter('regions', df)
        rc.execute("regions = GenomicRegionCollection(hg38, regions)")
        rc.execute("result = regions.scan_motif(motifs)")
        return pd.concat(rc.gather('result'))


def generate_dataset(genome, df):
    """Generate a dataset from a GenomicRegionCollection
    """
    gr = GenomicRegionCollection(hg38, df)
    # 1. Generate sequences
    input = [s.one_hot for s in gr.center_expand(1000).collect_sequence()]
    # 2. Generate motif scores
    target = pd.concat([parallel_scan_NrMotifV1(partition_df) for chrom, partition_df in tqdm(gr.center_expand(1000).as_df().drop('genome', axis=1).groupby('Chromosome'))])
    return input, target


# %%
def save_sparse_pandas_df(df, filename):
    data = df.sparse.to_coo()
    columns = df.columns
    index = df.index.to_numpy()
    save_npz(filename, data)
    np.save(filename + '.columns.npy', columns)
    np.save(filename + '.index.npy', index)
    return


def load_sparse_pandas_df(filename):
    data = load_npz(filename+'.npz')
    columns = np.load(filename + '.columns.npy', allow_pickle=True)
    index = np.load(filename + '.index.npy', allow_pickle=True)
    return pd.DataFrame.sparse.from_spmatrix(data, index=index, columns=columns)


def save_input(input, filename):
    save_npz(filename, vstack(input))
    return


def save_dataset(input, target, filename):
    save_input(input, filename + '_data')
    save_sparse_pandas_df(target, filename + '_target')
    return


def load_dataset(filename):
    target = load_sparse_pandas_df(filename + '_target')
    n_samples = target.shape[0]
    data = load_npz(filename + '_data'+'.npz')
    seq_len = data.shape[0]//n_samples
    return [data[i*seq_len:(i+1)*seq_len] for i in range(n_samples)], target


# %%
atac = read_bed("../test/test.atac.bed").as_df()
atac_chr1 = atac[atac.Chromosome != 'chr8']
atac_chr2 = atac[atac.Chromosome == 'chr8']
#%%
train_data, train_target = generate_dataset(hg38, atac_chr1)
test_data, test_target = generate_dataset(hg38, atac_chr2)
# %%
save_dataset(train_data, train_target, 'train')
save_dataset(test_data, test_target, 'test')
# %%
train_data, train_target = load_dataset('train')
test_data, test_target = load_dataset('test')
# %%
