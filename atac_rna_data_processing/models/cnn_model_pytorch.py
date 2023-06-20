#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Module, Linear
import os
import random as rn
from torch.nn import MSELoss
from torch.optim import Adam
from collections import OrderedDict
#%%
os.environ['PYTHONHASHSEED'] = '0'

#%% Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)    
    torch.manual_seed(seed)
    rn.seed(seed)

#%% Cropping layer
class Cropping1D(nn.Module):
    def __init__(self, crop_size):
        super(Cropping1D, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, self.crop_size:-self.crop_size]

#%% BPNet
class BPNet(Module):
    def __init__(self, filters, n_dil_layers, sequence_len, out_pred_len):
        super(BPNet, self).__init__()

        self.conv1 = Conv1d(4, filters, 21, padding=10)
        self.relu = nn.ReLU()

        self.convs = nn.ModuleList()
        self.crops = nn.ModuleList()

        for i in range(1, n_dil_layers + 1):
            self.convs.append(Conv1d(filters, filters, 3, dilation=2**i, padding=2**(i-1)))
            self.crops.append(Cropping1D((sequence_len - 2**i) // 2))

        self.conv2 = Conv1d(filters, 1, 1, padding=37)
        self.crop2 = Cropping1D(int((sequence_len - 75) / 2) - int(out_pred_len / 2))

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Always reduce sequence length to 1

        #self.avg_pool = nn.AvgPool1d(filters)
        self.dense = Linear(filters, 1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        print(f"Initial shape: {x.shape}")

        for i, conv in enumerate(self.convs):
            if x.shape[2] < 3:  # stop if the sequence length becomes less than the kernel size
                break
            x = conv(x)
            crop_size = 2**(i+1)  # Adjust the crop size
            if crop_size < x.shape[2]:  # Only perform cropping if crop_size is less than the current sequence length
                x = x[:, :, crop_size:-crop_size]  # Crop here instead of in a separate layer
            print(f"Shape after conv and crop: {x.shape}")


        profile_out_precrop = self.conv2(x)
        print(f"Shape after conv2: {profile_out_precrop.shape}")

        profile_out = profile_out_precrop.view(profile_out_precrop.shape[0], -1)
        print(f"Shape after view: {profile_out.shape}")

        x_pooled = self.avg_pool(x)
        print(f"Shape after avg_pool: {x_pooled.shape}")

        x_flattened = x_pooled.view(x_pooled.shape[0], -1)
        
        count_out = self.dense(x_flattened)

        return profile_out, count_out

#%% Get Model
def get_model(args, model_params):
    # Seed initialization
    set_seed(args.seed)
    
    assert("bias_model_path" in model_params.keys())  # bias model path not specfied for model
    filters=int(model_params['filters'])
    n_dil_layers=int(model_params['n_dil_layers'])
    counts_loss_weight=float(model_params['counts_loss_weight'])
    sequence_len=int(model_params['inputlen'])
    out_pred_len=int(model_params['outputlen'])
    
    # Your load_pretrained_bias function was not included in this example. Here we just load the model.
    bias_model = torch.load(model_params['bias_model_path'])
    bias_model.eval()  # Ensure the model is in evaluation mode

    bpnet_model = BPNet(filters, n_dil_layers, sequence_len, out_pred_len)
    
    model = nn.ModuleList([bias_model, bpnet_model])
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()  # Replace with appropriate loss function
    
    return model, optimizer, criterion

# %%
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = nn.Conv1d(4, 1, 21, padding='valid')

    def forward(self, x):
        profile_out = self.conv(x).view(x.shape[0], -1)
        count_out = torch.mean(x, dim=2, keepdim=True)  # some dummy operation for count_out
        return profile_out, count_out

#%% Save the dummy model
dummy_model = DummyModel()
torch.save(dummy_model, "dummy_model.pth")

# %% Test the model
class Args:
    def __init__(self, learning_rate, seed):
        self.learning_rate = learning_rate
        self.seed = seed

# Use the path to your actual bias model
bias_model_path = "dummy_model.pth"

model_params = {
    'bias_model_path': bias_model_path,
    'filters': 32,
    'n_dil_layers': 6,
    'inputlen': 1000,
    'outputlen': 100,
    'counts_loss_weight': 1.0

}

args = Args(learning_rate=0.001, seed=12345)
#model, optimizer, criterion = get_model(args, model_params)
model = BPNet(filters=32, n_dil_layers=6, sequence_len=1000, out_pred_len=100)

# Then, to test the model, we can input a sample tensor.
# Assuming that the model takes input of shape (batch_size, channels, sequence_len)

input_tensor = torch.randn((10, 4, 1000)) # batch size is 10
profile_out, count_out = model(input_tensor)

print(profile_out.shape)  # should be [10, 100]
print(count_out.shape)  # should be [10, 1]


# %%
