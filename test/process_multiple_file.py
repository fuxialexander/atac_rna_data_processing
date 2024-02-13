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
import csv
import glob

# %% Define the directory containing your .atac.bed files
bed_files_directory = "/manitou/pmg/projects/resources/get_interpret/pretrain_human_bingren_shendure_apr2023/fetal_adult/"

# %% Use glob to find all .atac.bed files in the directory
bed_files = glob.glob(os.path.join(bed_files_directory, "*.atac.bed"))

# %% Define a function to process a single BED file
for bed_file in bed_files:
    # Extract base name for identifying files
    base_name = os.path.splitext(os.path.basename(bed_file))[0]
    base_name = base_name.replace('.atac', '')  # Ensure '.atac' is removed from the base name

    
    # Process each BED file
    df = read_bed(bed_file)
    df = df.df.rename(columns={"Name": "Score"})
    
    hg38 = Genome("hg38", "/manitou/pmg/users/hvl2108/atac_rna_data_processing/test/hg38.fa")  # Adjust this path as necessary
    gr = GenomicRegionCollection(df=df, genome=hg38)

    # Collect sequence, save data to Zarr
    data_zarr_output_path = os.path.join(bed_files_directory, f"{base_name}.zarr")
    gr.collect_sequence(target_length=1024).save_zarr(data_zarr_output_path)
    
    # Load the processed data
    data = zarr.load(data_zarr_output_path)['arr_0']

    # Load BED, extract target values
    bed_df = pd.read_csv(bed_file, sep='\t', header=None)
    target_values = bed_df[3].values

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, target_values, test_size=0.2, random_state=42)

    # Save the training data
    train_data_path = os.path.join(bed_files_directory, f"{base_name}_train.zarr")
    train_data = zarr.open(train_data_path, mode='w')
    train_data.array('arr_0', X_train)

    # Save the training target values, adjusted naming convention here
    train_target_values_path = os.path.join(bed_files_directory, f"train_target_values_{base_name}.zarr")
    train_target_values = zarr.open(train_target_values_path, mode='w')
    train_target_values.array('arr_0', y_train)

    # Save the validation data
    val_data_path = os.path.join(bed_files_directory, f"{base_name}_val.zarr")
    val_data = zarr.open(val_data_path, mode='w')
    val_data.array('arr_0', X_val)

    # Save the validation target values, adjusted naming convention here
    val_target_values_path = os.path.join(bed_files_directory, f"val_target_values_{base_name}.zarr")
    val_target_values = zarr.open(val_target_values_path, mode='w')
    val_target_values.array('arr_0', y_val)

    print(f"Processed and saved training and validation data and targets for {base_name}")

# %% 
class ZarrDataset(Dataset):
    def __init__(self, file_paths, target_file_paths, n):
        
        self.file_paths = file_paths
        self.target_file_paths = target_file_paths
        self.n = n
        
        # Get the lengths of each file and the cumulative lengths
        self.lengths = [zarr.open(file, mode='r')['arr_0'].shape[0] for file in file_paths]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)[:-1]

    def __len__(self):
        
        return sum(self.lengths) // self.n

    def __getitem__(self, idx):
        
        file_idx = np.searchsorted(self.cumulative_lengths, idx * self.n, side='right') - 1

        # Adjust idx to be the index within the file, referring to chunks, not individual samples
        if file_idx > 0:
            idx = idx - (self.cumulative_lengths[file_idx] // self.n)
        
        # Calculate start and end indices for the data
        start_idx = idx * self.n
        end_idx = start_idx + self.n

        # Load the chunk of data on-the-fly
        data = zarr.open(self.file_paths[file_idx], mode='r')['arr_0'][start_idx:end_idx]
        target_values = zarr.open(self.target_file_paths[file_idx], mode='r')['arr_0'][start_idx:end_idx]

        # Convert to tensors
        instances = torch.from_numpy(data)
        targets = torch.from_numpy(target_values)

        return instances, targets
    
    def print_cumulative_lengths(self):
        print("Cumulative lengths of datasets:", self.cumulative_lengths)

    def get_file_origin(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        return self.file_paths[file_idx]    

# %%

# Assuming bed_files_directory is where your Zarr files are stored
bed_files_directory = "/manitou/pmg/projects/resources/get_interpret/pretrain_human_bingren_shendure_apr2023/fetal_adult/"

# Use glob to find all data Zarr files and target Zarr files
data_zarr_files = glob.glob(os.path.join(bed_files_directory, "*_train.zarr"))
target_zarr_files = glob.glob(os.path.join(bed_files_directory, "train_target_values_*.zarr"))

# Ensure the lists are sorted to maintain matching order between data and target files
data_zarr_files.sort()
target_zarr_files.sort()

# Initialize ZarrDataset with the found file paths
dataset = ZarrDataset(data_zarr_files, target_zarr_files, 64)

# Initialize DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10)

# Iterate through dataloader batches for demonstration
num_batches_to_test = 3  # Number of batches to test

for batch_idx, (instances, targets) in enumerate(dataloader):
    if batch_idx >= num_batches_to_test:
        break  # Stop the loop after processing the specified number of batches

    print(f"Batch {batch_idx + 1}")
    print(f"Instances shape: {instances.shape}, Targets shape: {targets.shape}")
    # Here you can perform operations with 'instances' and 'targets'

# Example usage:
#%%
# file_paths = ['31.zarr', '184.zarr', '215.zarr']
# target_file_paths = ['train_target_values_31.zarr', 'train_target_values_184.zarr', 'train_target_values_215.zarr']

# # %%
# dataset =  ZarrDataset(file_paths, target_file_paths, 64)

# # %%
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10)

# #%%
# num_batches_to_test = 3  # Number of batches to test

# for batch_idx, (instances, targets) in enumerate(dataloader):
#     if batch_idx >= num_batches_to_test:
#         break  # Stop the loop after processing the specified number of batches

#     print(f"Batch {batch_idx + 1}")
#     print(f"Instances shape: {instances.shape}, Targets shape: {targets.shape}")
#     # Perform operations with 'instances' and 'targets'
#     # For example, pass them through a neural network, calculate loss, etc.


# %%

