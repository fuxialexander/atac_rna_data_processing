#%% Import package
import sys
sys.path.append('..')
from atac_rna_data_processing.io.region import *
import pandas as pd
from pyranges import read_bed
import zarr
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import wandb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# %% Initialize a new run
run_name = wandb.util.generate_id()
wandb.init(project="my-project", name=run_name)
config = wandb.config
config.batch_size = 64
config.epochs = 5
config.learning_rate = 0.0005

# %% Load data
df = read_bed("./k562_cut0.03.atac.bed")
df = df.df.rename(columns={"Name":"Score"})
df.head()

# %% Create a GenomicRegionCollection object
hg38 = Genome("hg38","/manitou/pmg/users/hvl2108/atac_rna_data_processing/test/hg38.fa")
gr = GenomicRegionCollection(df=df,genome=hg38)

# %% Create a Dataset object
gr.collect_sequence(target_length=1024).save_zarr("./test.zarr")

#%%
data = zarr.load('test.zarr')

# %%  Load the BED file as a pandas DataFrame
bed_df = pd.read_csv('k562_cut0.03.atac.bed', sep='\t', header=None)

# Extract the fourth column as your target values
target_values = bed_df[3].values

# %% Get the first instance
class ZarrDataset(Dataset):

    def __init__(self, file_path, target_file_path, sequence_length):
        
        raw_data = zarr.open(file_path, mode='r')['arr_0']
        self.data = torch.from_numpy(np.array(raw_data))
        
        raw_targets = zarr.open(target_file_path, mode='r')['arr_0']
        self.target_values = torch.from_numpy(np.array(raw_targets))
        
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start_position = idx * self.sequence_length
        end_position = start_position + self.sequence_length
        
        sequence_data = self.data[start_position:end_position]
        sequence_targets = self.target_values[start_position:end_position]
        
        return sequence_data, sequence_targets

#%%
# 1 region -> N regions
# CNN will process the N regions DNA sequences (N, L, 4) to (N, L/2, 128) (N, L/4, 128) -> sum(axis=1) -> (N, D)
# (N,D) put into a transformer

# %% Model Building
class ConvBlock(nn.Module):
    def __init__(self, size, stride=1, dilation=1, hidden_in=64, hidden=64, first_block=False):
        super(ConvBlock, self).__init__()
        
        # Inline padding calculation
        pad_len = (size - 1) * dilation // 2 if first_block else (size - 1) // 2
        
        layers = [
            nn.Conv1d(hidden_in, hidden, size, stride, pad_len, dilation=dilation),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        ]
        
        if not first_block:
            layers.extend([
                nn.Conv1d(hidden, hidden, size, padding=pad_len),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, size, padding=pad_len),
                nn.BatchNorm1d(hidden)
            ])
            # Inline addition of residual layer
            self.match_channels = nn.Conv1d(hidden_in, hidden, 1)
        else:
            self.match_channels = None
        
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block(x)
        if self.match_channels:
            x += self.match_channels(x)         
        return self.relu(x)

# %%
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_encoder_layers=1):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        x = x.permute(1, 0, 2) 
        x = self.transformer_encoder(x)
        return x.permute(1, 0, 2)
    

class CombinedBlock(nn.Module):
    def __init__(self, conv_params, transformer_params):
        super(CombinedBlock, self).__init__()

        # Conv Params
        size = conv_params.get('size', 3)
        stride = conv_params.get('stride', 1)
        dilation = conv_params.get('dilation', 1)
        hidden_in = conv_params.get('hidden_in', 64)
        hidden = conv_params.get('hidden', 64)
        first_block = conv_params.get('first_block', False)

        # Transformer Params
        d_model = transformer_params.get('d_model', 64)
        nhead = transformer_params.get('nhead', 8)
        num_encoder_layers = transformer_params.get('num_encoder_layers', 1)

        self.conv_block = ConvBlock(size, stride, dilation, hidden_in, hidden, first_block)
        self.transformer_block = TransformerBlock(d_model, nhead, num_encoder_layers)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.transformer_block(x)
        return x
