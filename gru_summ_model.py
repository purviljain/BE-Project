from torch import nn
import pandas as pd
import h5py
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SummarizationModel(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        super().__init__()

        # self.vgg - not needed at this stage because the dataset is already processed in that format

        # GRU layer

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)    

        # Multi layer percerptron with 2 layers

        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)

        self.mlp2 = nn.Linear(hidden_dim, 1)

    def forward(self, feature):

        # input - [frames, 1024]

        # print(feature.shape)
        feature = feature.unsqueeze(0)
        # feature - [1,frames,1024]

        # print(feature.shape)
        x, hidden = self.gru(feature)

        # [1, frames, 512]

        # print(x.shape)
        # removed ReLu
        x = self.mlp1(x)

        x = self.mlp2(x)

        # [1, frames, 1]

        # print(x.shape)
        output = x.view(-1, 1)
        output = output.T

        # print(output.shape)
        # [1,frames]

        return output
    
def get_model():
    
    INPUT_DIM = 1024
    HIDDEN_DIM = 512
    
    return SummarizationModel(HIDDEN_DIM, INPUT_DIM)
   