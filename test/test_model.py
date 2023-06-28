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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# %%
df = read_bed("./k562_cut0.05.atac.bed")
df = df.df.rename(columns={"Name":"Score"})
# %%
df.head()
# %%
#hg38 = Genome("hg38","/home/ubuntu/atac_rna_data_processing/test/hg38.fa")
hg38 = Genome("hg38","/home/tommy/atac_rna_data_processing/test/hg38.fa")
gr = GenomicRegionCollection(df=df,genome=hg38)

# %%
gr.collect_sequence(target_length=1000).save_zarr("./test.zarr")

#%%
data = zarr.load('test.zarr')

# %%  Load the BED file as a pandas DataFrame
bed_df = pd.read_csv('k562_cut0.05.atac.bed', sep='\t', header=None)

# Extract the fourth column as your target values
target_values = bed_df[3].values

# %% Get the first instance
class ZarrDataset(Dataset):

    def __init__(self, file_path, target_values):
        self.data = zarr.open(file_path, mode='r')['arr_0']
        self.target_values = target_values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = torch.from_numpy(self.data[idx])
        target = self.target_values[idx]  # Get corresponding target value
        return instance, target

# %% Assuming target_values is a list or array containing your target values
dataset = ZarrDataset('test.zarr', target_values)

# %% Debugging
instance, target = dataset[0]
print('Instance:', instance)
print('Target:', target)
print(f'Length of dataset: {len(dataset)}')

# %% ConvBlock method
class ConvBlock(nn.Module):
    def __init__(self, size, stride = 2, hidden_in = 64, hidden = 64):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len, dilation=2),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.relu = nn.ReLU()
        # Add a fully connected layer at the end
        self.fc = nn.Linear(4000, 1)

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        x = x.reshape(x.size(0), -1) # Flatten the tensor
        # apply the fully connected layer at the end
        x = self.fc(x)
        return x

# Define device
device = torch.device('cuda')

# %% Split into train and validation sets
# Compute the index where to split the data
split_idx = int(len(data['arr_0']) * 0.8)  # 80% for training

# Create the training data
train_data = zarr.open('train.zarr', mode='w')
train_data.array('arr_0', data['arr_0'][:split_idx])

# Create the validation data
val_data = zarr.open('val.zarr', mode='w')
val_data.array('arr_0', data['arr_0'][split_idx:])

# Split the target values into training and validation sets
target_values_train = target_values[:split_idx]
target_values_val = target_values[split_idx:]

# Create data loaders
train_loader = DataLoader(ZarrDataset('train.zarr', target_values_train), batch_size=4, shuffle=True, num_workers=10)
val_loader = DataLoader(ZarrDataset('val.zarr', target_values_val), batch_size=4, shuffle=True, num_workers=10)

# %% Define loss function and optimizer
model = ConvBlock(size=3, stride=2, hidden_in=4, hidden=32).cuda()  # Update the model
criterion = nn.MSELoss().cuda() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% Train model
for epoch in range(5):  # Number of epochs
    model.train()
    
    # Start timer
    start_time = time.time()

    # Create progress bar with tqdm
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    train_loss = 0
    
    for i, batch in progress_bar:
        seq, labels = batch
        seq = seq.transpose(1,2).float().cuda()
        labels = labels.float().cuda()  # Assuming labels are float type. Adjust as necessary
        optimizer.zero_grad()
        outputs = model(seq)
        loss = criterion(outputs, labels)  # Compare model's output with target labels
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    # End timer
    end_time = time.time()

    # Compute elapsed time
    elapsed_time = end_time - start_time
    print(f"\nTime taken for epoch {epoch+1}: {elapsed_time:.2f} seconds. Average loss: {train_loss / len(train_loader)}")

# Close the progress bar after the loop ends
progress_bar.close()


# %%
