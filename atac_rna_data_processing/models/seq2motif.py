# this is a pytorch implementation of the seq2motif model
# input: one hot encoded sequence of 1000 bp
# output: a 282 dimension vector, which is the normalized binding score of each motif
# architecture: 1D convolutional neural network with attention pooling, 5 CNN layers, 1 multi-head attention layer, 1 fully connected layer, with ReLU activation function for all layers
# loss: BCEWithLogitsLoss
# optimizer: Adam
# learning rate: 0.001
# the model is trained on the 282 motifs from the motif clustering database

# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=embedding_dim, kernel_size=19, stride=1, padding=9)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=11, stride=1, padding=5)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1)
        self.attention = MultiHeadAttention(input_dim=embedding_dim, output_dim=embedding_dim, num_heads=num_heads)
        self.fc1 = nn.Linear(embedding_dim, num_motifs)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x: (batch_size, seq_len, 4)
        print(x.size)
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = self.dropout(F.relu(self.conv4(x)))
        x = self.dropout(F.relu(self.conv5(x)))
        x = self.attention(x)
        x = self.fc1(x)
        return x
    

