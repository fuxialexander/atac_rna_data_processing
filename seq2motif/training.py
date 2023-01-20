# this file generate training data for seq2motif model
# specifically:
# 1. Given a list of regions and a genome assembly, generate a list of sequences (one-hot encoded numpy array) as inputs
# 2. Given a list of sequences, generate a list of motif scores as targets
#%%
import sys; sys.path.append('..')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, csr_matrix, load_npz, save_npz, vstack
from tqdm import tqdm
from atac_rna_data_processing.models.enformer import Seq2Motif
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
#%%
train_data, train_target = load_dataset('train')
test_data, test_target = load_dataset('test')

train_target = np.log10(train_target+1)
test_target = np.log10(test_target+1)

# %%
# train one epoch
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        # evaluate every 100 batches
        if batch_idx % 100 == 0:
            test_loss = evaluate(model, test_loader, criterion, device)
            print('test_loss: ', test_loss)
            
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)['motif']
        loss = criterion(output.flatten(), target.to_dense().flatten())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader.dataset)
    return train_loss

# evaluate function
def evaluate(model, test_loader, criterion, device, return_output=False):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)['motif']
            test_loss += criterion(output.flatten(), target.to_dense().flatten()).item()
    test_loss /= len(test_loader.dataset)
    if return_output:
        return test_loss, output.cpu().numpy(), target.to_dense().cpu().numpy()
    return test_loss
#%%
# dataloader
class MotifDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = csr_matrix(target.sparse.to_dense())
        self.label = target.index.values
        self.motifs = target.columns.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].toarray().T, self.target[idx]

def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)
    
def sparse_batch_collate(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, targets_batch = zip(*batch)
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch)
    return data_batch, targets_batch



#%%
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set random seed
torch.manual_seed(0)
np.random.seed(0)
# set hyperparameters
num_motifs = 282
seq_len = 1000
embedding_dim = 128
num_heads = 4
num_layers = 1
dropout_rate = 0.5
batch_size = 256
num_epochs = 100
learning_rate = 0.0005
# build dataloader
train_dataset = MotifDataset(train_data, train_target)
test_dataset = MotifDataset(test_data, test_target)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sparse_batch_collate,drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=sparse_batch_collate,drop_last=True)
#%%
model = Seq2Motif().to(device)
# set optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

#%%
# train model
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    test_loss = evaluate(model, test_loader, criterion, device)
    train_loss = train(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print('Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
# save model
torch.save(model.state_dict(), 'model.pth')
# save losses
np.save('train_losses.npy', np.array(train_losses))
np.save('test_losses.npy', np.array(test_losses))



# %%
loss, target, output = evaluate(model, test_loader, criterion, device, return_output=True)
# %%
import seaborn as sns

sns.scatterplot(x=target[:, 0].reshape(-1), y=output[:, 0].reshape(-1), hue=np.repeat(train_dataset.motifs, 256, axis=0).reshape(-1, 256).T[:, 0].reshape(-1), s=10, alpha=1, legend=False)
# %%
torch.save(model.state_dict(), 'model.pth')
# %%
