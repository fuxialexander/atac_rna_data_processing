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
import logging
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import wandb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# %% Initialize a new run
# generate a new run name
run_name = wandb.util.generate_id()
wandb.init(project="my-project", name=run_name)
config = wandb.config
config.batch_size = 16
config.epochs = 5
config.learning_rate = 0.0001

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

    # def __init__(self, file_path, target_values):
    #     self.data = torch.from_numpy(np.array(zarr.open(file_path, mode='r')['arr_0']))
    #     self.target_values = target_values

    def __init__(self, file_path, target_file_path):
        self.data = torch.from_numpy(np.array(zarr.open(file_path, mode='r')['arr_0']))
        self.target_values = torch.from_numpy(np.array(zarr.open(target_file_path, mode='r')['arr_0']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        target = self.target_values[idx]  # Get corresponding target value
        return instance, target

# %% ConVBLock method 2
class FirstConvBlock(nn.Module):
    def __init__(self, size, stride = 2, hidden_in = 64, hidden = 64):
        super(FirstConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len, dilation=2),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        out = self.relu(scaled)
        return out

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

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out

# Define device
device = torch.device('cuda')

# Define model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.block1 = FirstConvBlock(size=20, stride=1, hidden_in=4, hidden=128)
        self.maxpool1 = nn.AvgPool1d(20)
        self.dropout1 = nn.Dropout(0.5)  # 50% dropout

        self.block2 = ConvBlock(size=3, stride=2, hidden_in=128, hidden=128)
        self.maxpool2 = nn.AvgPool1d(2)
        self.dropout2 = nn.Dropout(0.5)  # 50% dropout

        self.block3 = ConvBlock(size=3, stride=2, hidden_in=128, hidden=128)
        self.maxpool3 = nn.AvgPool1d(2)
        
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.block3(x)
        x = self.maxpool3(x)
        x = x.sum(dim=2)
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x.squeeze()
    
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

# Create the training data
# train_data = zarr.open('train.zarr', mode='w')
# train_data.array('arr_0', data['arr_0'][:split_idx])

# # Create the validation data
# val_data = zarr.open('val.zarr', mode='w')

# val_data.array('arr_0', data['arr_0'][split_idx:])

# # Split the target values into training and validation sets
# target_values_train = target_values[:split_idx]
# target_values_val = target_values[split_idx:]

# # %% Change batch size to 256
# batch_size = 256
# train_loader = DataLoader(ZarrDataset('train.zarr', target_values_train), batch_size=batch_size, shuffle=True, num_workers=10)
# val_loader = DataLoader(ZarrDataset('val.zarr', target_values_val), batch_size=batch_size, shuffle=True, num_workers=10)

# %% Define loss function and optimizer
# model = ConvBlock(size=3, stride=2, hidden_in=4, hidden=32).cuda()  # Update the model
model = MyModel().cuda()
criterion = nn.MSELoss().cuda() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# Specify the path to your checkpoint
checkpoint_path = 'checkpoint_path.pth'

if os.path.isfile(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Restore the state of model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")

else:
    print("No checkpoint found at specified path, training from scratch.")
    start_epoch = 0

# %% Set up the logger
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(message)s')

# %% Evaluate log loss
def evaluate(model, criterion, data_loader):
    model.eval()  # set the model to evaluation mode
    total_loss = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
        for batch in data_loader:
            seq, labels = batch
            seq = seq.transpose(1,2).float().cuda()
            labels = labels.float().cuda()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # convert lists to numpy arrays
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    # calculate additional metrics
    pearson_corr, _ = pearsonr(all_outputs.flatten(), all_labels.flatten())
    r2 = r2_score(all_labels, all_outputs)

    return total_loss / len(data_loader), pearson_corr, r2

# %% Train model
for epoch in range(100):  # Number of epochs
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
        # Log the training loss for each batch
        wandb.log({"Batch Training Loss": loss.item()})

    # Call evaluate at the end of each epoch for both training and validation sets
    train_loss, train_pearson_corr, train_r2 = evaluate(model, criterion, train_loader)
    val_loss, val_pearson_corr, val_r2 = evaluate(model, criterion, val_loader)
    # Calculate the average training loss for this epoch
    avg_train_loss = train_loss / len(train_loader)
    # Log the losses, additional metrics and the current epoch to wandb
    wandb.log({
        "Epoch": epoch, 
        "Average Training Loss": avg_train_loss, 
        "Training Pearson Correlation": train_pearson_corr, 
        "Training R-squared": train_r2,
        "Validation Loss": val_loss, 
        "Validation Pearson Correlation": val_pearson_corr, 
        "Validation R-squared": val_r2
    }) 

    # End timer
    end_time = time.time()

    # Save a checkpoint after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_{epoch}.pth')


    # Compute elapsed time
    elapsed_time = end_time - start_time
    print(f"\nTime taken for epoch {epoch+1}: {elapsed_time:.2f} seconds. Average loss: {train_loss / len(train_loader)}")

# Close the progress bar after the loop ends
progress_bar.close()
