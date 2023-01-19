# this file generate training data for seq2motif model
# specifically:
# 1. Given a list of regions and a genome assembly, generate a list of sequences (one-hot encoded numpy array) as inputs
# 2. Given a list of sequences, generate a list of motif scores as targets
#%%
import pandas as pd
import sys; sys.path.append('..')
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from atac_rna_data_processing.models.seq2motif import Seq2Motif

import numpy as np
from scipy.sparse import load_npz, save_npz, vstack, coo_matrix, csr_matrix
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
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.to_dense())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader.dataset)
    return train_loss

# evaluate function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target.to_dense()).item()
    test_loss /= len(test_loader.dataset)
    return test_loss



# dataloader
class MotifDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target.sparse.to_coo().tocsr()
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
batch_size = 128
num_epochs = 2
learning_rate = 0.001
# build dataloader
train_dataset = MotifDataset(train_data, train_target)
test_dataset = MotifDataset(test_data, test_target)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sparse_batch_collate,drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=sparse_batch_collate,drop_last=True)
#%%
# build model

# attention pooling layer
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # attention: (batch_size, seq_len, 1)
        attention = self.softmax(self.fc2(F.relu(self.fc1(x))))
        # output: (batch_size, input_dim)
        output = torch.sum(attention * x, dim=1)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # attention: (batch_size, seq_len, 1)
        attention = self.softmax(self.fc2(F.relu(self.fc1(x))))
        # output: (batch_size, input_dim)
        output = torch.sum(attention * x, dim=1)
        return output
    
# seq2motif model
class Seq2Motif(nn.Module):
    def __init__(self, num_motifs, seq_len, embedding_dim, num_heads, num_layers, dropout_rate):
        super(Seq2Motif, self).__init__()
        self.num_motifs = num_motifs
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=embedding_dim, kernel_size=20, stride=1, padding=10)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=10, stride=5, padding=5)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=10, stride=5, padding=5)
        self.conv4 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=5, stride=5, padding=2)
        self.conv5 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=5, stride=5, padding=1)
        # self.attention = MultiHeadAttention(input_dim=embedding_dim, output_dim=embedding_dim, num_heads=num_heads)
        self.fc1 = nn.Linear(embedding_dim*2, num_motifs)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x: (batch_size, seq_len, 4)
        
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = self.dropout(F.relu(self.conv4(x)))
        x = self.dropout(F.relu(self.conv5(x)))
        # x = self.attention(x)
        x = self.fc1(x.to_dense().view(batch_size, -1))
        return x
    

model = Seq2Motif(num_motifs, seq_len, embedding_dim, num_heads, num_layers, dropout_rate).to(device)
# set optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

#%%

# train model
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss = evaluate(model, test_loader, criterion, device)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print('Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
# save model
torch.save(model.state_dict(), 'model.pth')
# save losses
np.save('train_losses.npy', np.array(train_losses))
np.save('test_losses.npy', np.array(test_losses))



# %%
