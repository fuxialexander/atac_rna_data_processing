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

#% 
class ZarrDataset(Dataset):
    # Removed sequence_length from the constructor
    def __init__(self, file_path, target_file_path):
        self.data = torch.from_numpy(zarr.open(file_path, mode='r')['arr_0'][:])
        self.target_values = torch.from_numpy(zarr.open(target_file_path, mode='r')['arr_0'][:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target_values[idx]
        data = data.permute(1, 0)  
        return data, target
    
#%
print(data['arr_0'].shape)

# %% Split the data into training and validation sets
device = torch.device('cuda')  # Usually, the device is referred to as 'cuda' for NVIDIA GPUs

X_train, X_val, y_train, y_val = train_test_split(data['arr_0'], target_values, test_size=0.2, random_state=42)

train_data = zarr.open('train.zarr', mode='w')
train_data.array('arr_0', X_train)

train_target_values = zarr.open('train_target_values.zarr', mode='w')
train_target_values.array('arr_0', y_train)

val_data = zarr.open('val.zarr', mode='w')
val_data.array('arr_0', X_val)

val_target_values = zarr.open('val_target_values.zarr', mode='w')
val_target_values.array('arr_0', y_val)

train_loader = DataLoader(
    ZarrDataset('train.zarr', 'train_target_values.zarr'),
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=10,
    drop_last=True 

)

val_loader = DataLoader(
    ZarrDataset('val.zarr', 'val_target_values.zarr'),
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=10,
    drop_last=True  

)

# %%
for data, target in train_loader:
    print(f"Train batch - data shape: {data.shape}, target shape: {target.shape}")
    break  # Just to check the first batch, then exit the loop.

# Validate val_loader
for data, target in val_loader:
    print(f"Validation batch - data shape: {data.shape}, target shape: {target.shape}")
    break

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

class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_encoder_layers=1):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return x

# %% 
class CombinedBlock(nn.Module):
    def __init__(self, conv_size, conv_stride=1, conv_dilation=1, 
                 hidden_in=64, hidden=64, first_block=False, 
                 d_model=64, nhead=8, num_encoder_layers=1):
        super(CombinedBlock, self).__init__()
        
        assert hidden == d_model, "Output feature size of ConvBlock must match input feature size of TransformerBlock"

        self.conv_block = ConvBlock(conv_size, conv_stride, conv_dilation, hidden_in, hidden, first_block)
        self.transformer_block = TransformerBlock(d_model, nhead, num_encoder_layers)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)
        x = self.transformer_block(x)
        
        return x

# %%
class RegionalCombinedBlock(nn.Module):
    def __init__(self, num_regions, conv_size, conv_stride=1, conv_dilation=1, 
                 hidden_in=64, hidden=64, first_block=False, 
                 d_model=64, nhead=8, num_encoder_layers=1):
        super(RegionalCombinedBlock, self).__init__()

        # Create a list of CombinedBlocks for each region
        self.blocks = nn.ModuleList([
            CombinedBlock(conv_size, conv_stride, conv_dilation, hidden_in, hidden, first_block, d_model, nhead, num_encoder_layers)
            for _ in range(num_regions)
        ])

    def forward(self, x):
        # Assuming x is of shape (batch_size, channels, sequence_length)
        # We'll split the sequence length dimension into num_regions
        print("Shape of x before unpacking:", x.shape)
        batch_size, channels, seq_length = x.shape
        region_length = seq_length // len(self.blocks)
        outputs = []
        
        for i, block in enumerate(self.blocks):
            # Extract region from the input tensor
            start_idx = i * region_length
            end_idx = (i+1) * region_length
            region = x[:, :, start_idx:end_idx]       
            # Pass the region through the block
            out = block(region)
            
            outputs.append(out)
        
        # Concatenate the outputs along the sequence_length dimension
        return torch.cat(outputs, dim=2)



# # %%
# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming your model is composed of the RegionalCombinedBlock, which in turn includes ConvBlock and TransformerBlock
# Initialize your model
model = RegionalCombinedBlock(
    num_regions=10,  # Assuming 'N=10' from UnifiedModel(N=10) corresponds to num_regions
    conv_size=3,     # Example size, adjust as per your architecture's requirements
    conv_stride=1,
    conv_dilation=1,
    hidden_in=4,     # Assuming 4 channels for DNA sequence (A, T, G, C)
    hidden=64,       # Example feature size
    first_block=True,# Example, adjust as per your architecture's requirements
    d_model=64,
    nhead=8,
    num_encoder_layers=1
).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss().to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)

# %%
# Training loop
for epoch in range(config.epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Print loss every N batches
        if batch_idx % N == 0:
            print(f'Epoch {epoch} Batch {batch_idx} Loss {loss.item()}')

    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss}')



# # %%

# %%
