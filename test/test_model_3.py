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
# generate a new run name
run_name = wandb.util.generate_id()
wandb.init(project="seq_to_atac", name='version0.1-'+run_name)
config = wandb.config
config.batch_size = 64
config.epochs = 5
config.learning_rate = 0.0005

# %%
df = read_bed("/manitou/pmg/projects/resources/get_interpret/pretrain_human_bingren_shendure_apr2023/fetal_adult/118.atac.bed")
# df = read_bed("./k562_cut0.03.atac.bed")
df = df.df.rename(columns={"Name":"Score"})
# %%
df.head()
# %%
hg38 = Genome("hg38","/manitou/pmg/users/hvl2108/atac_rna_data_processing/test/hg38.fa")
gr = GenomicRegionCollection(df=df,genome=hg38)

# %%
gr.collect_sequence(target_length=1024).save_zarr("./test.zarr")

#%%
data = zarr.load('test.zarr')
#%%
data['arr_0'].shape
# %%  Load the BED file as a pandas DataFrame
bed_df = pd.read_csv("/manitou/pmg/projects/resources/get_interpret/pretrain_human_bingren_shendure_apr2023/fetal_adult/118.atac.bed", sep='\t', header=None)
# bed_df = pd.read_csv('k562_cut0.03.atac.bed', sep='\t', header=None)
# Extract the fourth column as your target values
target_values = bed_df[3].values

# %% Get the first instance
class ZarrDataset(Dataset):

    def __init__(self, file_path, target_file_path,n):
        self.data = torch.from_numpy(np.array(zarr.open(file_path, mode='r')['arr_0']))
        self.target_values = torch.from_numpy(np.array(zarr.open(target_file_path, mode='r')['arr_0']))
        self.n = n

    def __len__(self):
        return len(self.data) // self.n

    def __getitem__(self, idx):
        # instance = self.data[idx]
        # target = self.target_values[idx]  # Get corresponding target value
        start_idx = idx * self.n
        end_idx = start_idx + self.n

        instances = self.data[start_idx:end_idx]
        targets = self.target_values[start_idx:end_idx]  # Get corresponding target values for each region

        return instances, targets

#%%
# 1 region -> N regions
# CNN will process the N regions DNA sequences (N, L, 4) to (N, L/2, 128) (N, L/4, 128) -> sum(axis=1) -> (N, D)
# (N,D) put into a transformer
# %% Model Building
class ConvBlock(nn.Module):
    def __init__(self, size, stride=1, dilation=1, hidden_in=64, hidden=64, first_block=False):
        
        super(ConvBlock, self).__init__()
        
        # Calculate padding to maintain size
        pad_len = self._calculate_padding(size, dilation, first_block)
        
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
        
        self.block = nn.Sequential(*layers)
        self._add_residual_layers(hidden_in, hidden, first_block)
        self.relu = nn.ReLU()

    def _calculate_padding(self, size, dilation, first_block):
        if first_block:
            return (size - 1) * dilation // 2
        else:
            return (size - 1) // 2

    def _add_residual_layers(self, hidden_in, hidden, first_block):
        if not first_block:
            self.match_channels = nn.Conv1d(hidden_in, hidden, 1)
        else:
            self.match_channels = None

    def forward(self, x):
        
        x = self.block(x)

        if self.match_channels:  # Adjust channels if not a first block
            res = self.match_channels(x)
            x = res + x
            
        return self.relu(x)

#
class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=1):
        
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        x = self.transformer(x)
        return x


class UnifiedModel(nn.Module):
    def __init__(self, N, d_model=128, nhead=8):
        
        super(UnifiedModel, self).__init__()

        # CNN components
        self.cnn_blocks = self._create_cnn_blocks()

        # Transformer components
        self.N = N  # Number of regions
        self.transformer = TransformerBlock(d_model=d_model, nhead=nhead)
        self.fc_final = nn.Conv1d(d_model * 10, 1, 1)

    def _create_cnn_blocks(self):
        blocks = nn.Sequential(
            ConvBlock(size=16, stride=1, hidden_in=4, hidden=128, first_block=True, dilation=1),
            nn.MaxPool1d(16),
            ConvBlock(size=2, stride=1, hidden_in=128, hidden=128),
            nn.MaxPool1d(2),
            ConvBlock(size=2, stride=1, hidden_in=128, hidden=128),
            nn.MaxPool1d(2),
            ConvBlock(size=2, stride=1, hidden_in=128, hidden=128),
           
        )
        return blocks

    def cnn_forward(self, x):
        print("Shape of input to cnn_forward:", x.shape)
        x = self.cnn_blocks(x)
        return x

    def forward(self, x):
        
        # regions = x[:, :, :, :4 ]
        regions = [x[:, i, :, :] for i in range (self.N)]
        #regions = [x[:, :, i * split_size:(i + 1) * split_size] for i in range(self.N)]

        cnn_outs = [self.cnn_forward(region) for region in regions]
        print(cnn_outs[0].shape)
        #THIS IS REDUCED TO [B,R,L',D]
        cnn_out = torch.cat(cnn_outs, dim=2).transpose(1,2)#.unsqueeze(-1)  # Shape: (batch_size, feature_size, N*S)
        #INPUT RESHAPE to get to [B,R,-1]
        print("Shape of cnn_out:", cnn_out.shape)
        trans_out = self.transformer(cnn_out)
        B, NS, D = trans_out.shape # Shape: (batch_size, N*S, feature_size)
        S = NS // self.N
        trans_out = trans_out.reshape(B, self.N, S*D).transpose(1,2)  # Shape: (batch_size, feature_size, N)

        
        #MAKE SURE shape of fully connted layer is [B,R,-1] where -1 is L' * D
        output = self.fc_final(trans_out)
        print(output.shape)

        output = F.softplus(output)
        return output

# Define device
device = torch.device('cuda')
    
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


train_loader = DataLoader(ZarrDataset('train.zarr', 'train_target_values.zarr', 16), batch_size= 64, shuffle=True, num_workers=10, drop_last=True)
val_loader = DataLoader(ZarrDataset('val.zarr', 'val_target_values.zarr', n=16), batch_size= 64, shuffle=False, num_workers=10, drop_last=True)

# %% Define loss function and optimizer
model = UnifiedModel(N=16).cuda()
criterion = nn.MSELoss()#.cuda() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001)

# Specify the path to your checkpoint
checkpoint_path = 'checkpoint_path.pth'

if os.path.isfile(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Restore the state of model and optimizer
    model.load_state_dic(checkpoint['model_state_dict'])
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
        for batch in tqdm(data_loader):
            seq, labels = batch
            #print("Data shape before transpose method:", seq.size, labels.size)
            seq = seq.transpose(2,3).float().cuda()
            print(seq)
            #print("Data shape after transpose method:", seq.size, labels.size)
            labels = labels.float().cuda()
            print(seq.shape)
            outputs = model(seq).squeeze(1)
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
for epoch in range(50):  # Number of epochs
    model.train()
    
    # Start timer
    start_time = time.time()

    # Create progress bar with tqdm
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    train_loss = 0
    
    for i, batch in progress_bar:

        seq, labels = batch
        print("Data shape before transpose method:", seq.size, labels.size)
        seq = seq.transpose(2, 3).float().cuda()
        labels = labels.float().cuda()  # Assuming labels are float type. Adjust as necessary
        print("Shape of input: ", seq.shape)
        print("Shape of target labels:", labels.shape)
        
        optimizer.zero_grad()
        outputs = model(seq).squeeze(1)
        
        print("Shape of model output:", outputs.shape)
        
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

# %%
train_loss, train_pearson_corr, train_r2 = evaluate(model, criterion, train_loader)
# %%

