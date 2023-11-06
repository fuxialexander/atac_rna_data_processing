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
# class ZarrDataset(Dataset):

#     def __init__(self, file_path, target_file_path, sequence_length):
        
#         raw_data = zarr.open(file_path, mode='r')['arr_0']
#         self.data = torch.from_numpy(np.array(raw_data))
#         print("Shape after loading data:", self.data.shape)    
#         raw_targets = zarr.open(target_file_path, mode='r')['arr_0']
#         self.target_values = torch.from_numpy(np.array(raw_targets))
        
#         self.sequence_length = sequence_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         start_position = idx * self.sequence_length
#         end_position = start_position + self.sequence_length
        
#         sequence_data = self.data[start_position:end_position]
#         sequence_targets = self.target_values[start_position:end_position]
#         print("Shape of sequence_data:", sequence_data.shape)
#         return sequence_data, sequence_targets

class ZarrDataset(Dataset):
    # Removed sequence_length from the constructor
    def __init__(self, file_path, target_file_path):
        self.data = torch.from_numpy(zarr.open(file_path, mode='r')['arr_0'][:])
        self.target_values = torch.from_numpy(zarr.open(target_file_path, mode='r')['arr_0'][:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target_values[idx]
    
# %%
# Define device
device = torch.device('cuda')  # Usually, the device is referred to as 'cuda' for NVIDIA GPUs

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(data['arr_0'], target_values, test_size=0.2, random_state=42)

# Save the training data
train_data = zarr.open('train.zarr', mode='w')
train_data.array('arr_0', X_train)

# Save the training target values
train_target_values = zarr.open('train_target_values.zarr', mode='w')
train_target_values.array('arr_0', y_train)

# Save the validation data
val_data = zarr.open('val.zarr', mode='w')
val_data.array('arr_0', X_val)

# Save the validation target values
val_target_values = zarr.open('val_target_values.zarr', mode='w')
val_target_values.array('arr_0', y_val)

# Then proceed to create your dataloaders
train_loader = DataLoader(ZarrDataset('train.zarr', 'train_target_values.zarr'), batch_size=config.batch_size, shuffle=True, num_workers=10)
val_loader = DataLoader(ZarrDataset('val.zarr', 'val_target_values.zarr'), batch_size=config.batch_size, shuffle=False, num_workers=10)

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



# % 

# # %%
# # Define device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Assuming your model is composed of the RegionalCombinedBlock, which in turn includes ConvBlock and TransformerBlock
# # Initialize your model
# model = RegionalCombinedBlock(
#     num_regions=10,  # Assuming 'N=10' from UnifiedModel(N=10) corresponds to num_regions
#     conv_size=3,     # Example size, adjust as per your architecture's requirements
#     conv_stride=1,
#     conv_dilation=1,
#     hidden_in=4,     # Assuming 4 channels for DNA sequence (A, T, G, C)
#     hidden=64,       # Example feature size
#     first_block=True,# Example, adjust as per your architecture's requirements
#     d_model=64,
#     nhead=8,
#     num_encoder_layers=1
# ).to(device)

# # Define loss function and optimizer
# criterion = nn.MSELoss().to(device) 
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)

# # %% Training
# def evaluate(model, criterion, data_loader):
#     model.eval()  # set the model to evaluation mode
#     total_loss = 0
#     all_outputs = []
#     all_labels = []
#     with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
#         for batch in tqdm(data_loader):
#             seq, labels = batch
#             seq = seq.transpose(1, 3).transpose(2, 3).float().cuda()  # Transpose and move to GPU
#             labels = labels.float().cuda()  # Move to GPU
#             outputs = model(seq)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             all_outputs.extend(outputs.cpu().numpy())  # Move outputs back to CPU for numpy conversion
#             all_labels.extend(labels.cpu().numpy())    # Move labels back to CPU for numpy conversion

#     # convert lists to numpy arrays
#     all_outputs = np.array(all_outputs)
#     all_labels = np.array(all_labels)
#     # calculate additional metrics
#     pearson_corr, _ = pearsonr(all_outputs.flatten(), all_labels.flatten())
#     r2 = r2_score(all_labels, all_outputs)

#     return total_loss / len(data_loader), pearson_corr, r2

# # %% Main Training Loop
# for epoch in range(config.epochs):  # loop over the dataset multiple times
#     model.train()
    
#     # Initialize progress bar
#     progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
#     train_loss = 0
    
#     for i, batch in progress_bar:
#         # Prepare input and target labels
#         seq, labels = batch
#         seq = seq.transpose(1, 3).transpose(2, 3).float().to(device)
#         labels = labels.float().to(device)
        
#         # Zero the parameter gradients
#         optimizer.zero_grad()
        
#         # Forward + backward + optimize
#         outputs = model(seq)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         # Accumulate loss
#         train_loss += loss.item()
        
#         # Log the training loss for each batch
#         wandb.log({"Batch Training Loss": loss.item()})
    
#     # Evaluate on both training and validation sets after each epoch
#     train_loss, train_pearson_corr, train_r2 = evaluate(model, criterion, train_loader)
#     val_loss, val_pearson_corr, val_r2 = evaluate(model, criterion, val_loader)
    
#     # Calculate and log average losses and metrics
#     avg_train_loss = train_loss / len(train_loader)
#     wandb.log({
#         "Epoch": epoch,
#         "Average Training Loss": avg_train_loss,
#         "Training Pearson Correlation": train_pearson_corr,
#         "Training R-squared": train_r2,
#         "Validation Loss": val_loss,
#         "Validation Pearson Correlation": val_pearson_corr,
#         "Validation R-squared": val_r2
#     })
    
#     # Print epoch-wise average loss
#     print(f"Epoch {epoch + 1}/{config.epochs}. Average loss: {avg_train_loss:.4f}")

# # Close the wandb run after training
# wandb.finish()


# %%
