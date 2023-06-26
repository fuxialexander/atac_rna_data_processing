#%% Import package
import sys
sys.path.append('..')
from atac_rna_data_processing.io.region import *
import pandas as pd
from pyranges import read_bed
import zarr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# %%
# df = read_bed("./k562_cut0.05.atac.bed")
# df = df.df.rename(columns={"Name":"Score"})
# %%
# df.head()
# %%
# hg38 = Genome("hg38","/home/ubuntu/atac_rna_data_processing/test/hg38.fa")
# gr = GenomicRegionCollection(df=df,genome=hg38)
# %%
# gr.collect_sequence(target_length=1000).save_zarr("./test.zarr")

#%%
data = zarr.load('test.zarr')

# %% Get the keys as a list
keys = list(data.keys())
print('Keys in NPZ file:', keys)

# %% Get the first instance
class ZarrDataset(Dataset):
    def __init__(self, zarr_file):
        self.zarr_file = zarr_file
        self.data = zarr.open(self.zarr_file, mode='r')

    def __len__(self):
        # Assuming the first dimension is the number of instances
        return self.data['arr_0'].shape[0]

    def __getitem__(self, idx):
        # Retrieve the instance and convert to PyTorch tensor
        instance = self.data['arr_0'][idx]
        return torch.tensor(instance)

# %%
# Instantiate the ZarrDataset with the path to your Zarr file
dataset = ZarrDataset('test.zarr')

# %% Print the length of the dataset
print(f'Length of dataset: {len(dataset)}')

# %% Access a few instances in the dataset
for i in range(5):
    instance = dataset[i]
    print(f'Instance {i}: {instance.shape}')

# %%
class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16000, 4000)  # Assuming input length is 2000, pooling will reduce it to 1000
        #self.upsample = nn.Upsample(500)  # add an upsampling layer
        self.tconv = nn.ConvTranspose1d(10, 4, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        x = x.view(x.size(0), 10, -1)  # Reshape for the transposed convolution
        print(x.shape)
        x = self.tconv(x)
        print(x.shape)
        x = nn.functional.interpolate(x, size=(1000,), mode='linear', align_corners=False)
        return x


# %% Split into train and validation sets
# Compute the index where to split the data
split_idx = int(len(data['arr_0']) * 0.8)  # 80% for training
# Create the training data
train_data = zarr.open('train.zarr', mode='w')
train_data.array('arr_0', data['arr_0'][:split_idx])
# Create the validation data
val_data = zarr.open('val.zarr', mode='w')
val_data.array('arr_0', data['arr_0'][split_idx:])


# %% Create data loaders
train_loader = DataLoader(ZarrDataset('train.zarr'), batch_size=128, shuffle=True, num_workers=10)
val_loader = DataLoader(ZarrDataset('val.zarr'), batch_size=128, shuffle=True, num_workers=10)

# %% Define loss function and optimizer
model = Simple1DCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% Train model
for epoch in range(200):  # Number of epochs
    model.train()
    for i, batch in enumerate(train_loader):
        # Assuming your data is already in the correct shape (batch_size, channels, height, width)
        inputs = batch.float()
        inputs = inputs.permute(0, 2, 1)  # changes the order from [batch, length, channel] to [batch, channel, length]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Assuming it's an autoencoder, otherwise replace `inputs` with labels
        loss.backward()
        optimizer.step()

# %%
