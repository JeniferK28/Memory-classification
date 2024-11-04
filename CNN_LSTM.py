import torch.nn as nn
import torch
from torch import  Tensor
import numpy as np

device='cuda'

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(Net, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.conv1 = nn.Sequential(
            nn.Dropout2d(p=0.25),
            nn.Conv2d(1, 20, (124, 5),  padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
        )
        # Layer 2
        self.fc1 = nn.Linear(560, 150)
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True,dropout=0.25,bidirectional=False)
        self.linear = nn.Linear(hidden_dim, output_dim)




    def forward(self, x):
        out = self.conv1(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        #print(out.size())
        out_size = np.shape(out)
        #print(out.size())
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, out.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, out.size(0), self.hidden_dim).requires_grad_()
        h0, c0 = h0.to(device, dtype=torch.float), c0.to(device, dtype=torch.float)
        out = out.reshape(out_size[0], 1, out_size[1])
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out1, (hn, cn) = self.lstm1(out, (h0, c0))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out1 = self.linear(out1[:,0 , :])
        # out.size() --> 100, 10
        return out1



