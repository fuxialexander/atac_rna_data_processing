import pandas as pd
from scipy.sparse import load_npz, save_npz, vstack
import numpy as np

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

