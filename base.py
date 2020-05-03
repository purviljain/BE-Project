import json, pickle, os
import pandas as pd
from PIL import Image
import requests
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import torch
import gc, sys, psutil
import numpy as np
import random



class EncoderBase(nn.Module):
    
    def __init__(self, hidden_dim, input_dim, device):
        super().__init__()
        
        #self.lenet_model = self.get_lenet()
        
        self.device = device
        
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        
         
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mlp2 = nn.Linear(hidden_dim, 1)
        
        
    def get_cnn_features(self, frames):
    
        features = []
        for i in range(frames.shape[0]):

            frame = frames[i,:,:,:,:]

            cnn_model = models.googlenet(pretrained=True)
            lenet = nn.Sequential(*list(cnn_model.children())[:-2]).to(self.device)
            with torch.no_grad(): 
                x = lenet(frame)
                x = x.squeeze(3)
                x = x.squeeze(2)
            features.append(x)

        features = torch.stack(features, dim=0)

        return features.to(self.device)
        
    
#     def get_lenet(self):
        
#         model = models.googlenet(pretrained=True)
#         lenet = nn.Sequential(*list(model.children())[:-2])
        
#         #vgg.classifier = nn.Sequential(*[vgg.classifier[i] for i in range(4)])
        
#         return lenet
    
    def forward(self, x):
        
        # x = [bs, seq_len, 3, 224, 224] = [bs, 32, 3, 224, 224]
        
#         with torch.no_grad():
#             lenet_features = self.lenet_model(x)[:,:,-1,-1]

        lenet_features = self.get_cnn_features(x)

        
        # lenet_features = [bs, seq_len, 1024] or [bs, 32, 1024]
        #print('lenet_features: ',lenet_features.shape)
        
        outputs, hidden = self.gru(lenet_features)
        
        #print('output shape ',outputs.shape)
        #print('hidden shape ',hidden.shape)
        
        # outputs = [bs, seq_len, hidden_dim*num_directions] = [bs, 32, 512]
        # hidden = [num_layers*num_directions, bs, hidden_dim] = [1, bs, 512]
        summary_scores = self.mlp2(self.mlp1(outputs))
        
        summary_scores = summary_scores.view(-1,1).T
        # [1,frames] = [1,32]
        
        return hidden, summary_scores
     
        
        
        
class DecoderBase(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, output_vocab_dim, dropout):
        
        super().__init__()
        self.output_vocab_dim = output_vocab_dim
        
        self.embedding = nn.Embedding(num_embeddings=output_vocab_dim, embedding_dim=embedding_dim)
        
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim)
        
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_vocab_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inp, hidden):
        
        # inp = [bs] as decoder produces only one word per time-step
        # hidden = [num_layers, bs, hidden_dim]
        
        inp = inp.unsqueeze(0)
        
        # inp [1, bs], introduce the seq_len dimension which is 1 since 
        # only word is produced by the decoder in one fwd pass
        
        embed = self.dropout(self.embedding(inp))
        
        # embed = [1, bs, emb_dim]
        
        outputs, hidden = self.gru(embed, hidden)
        
        # outputs = [seq_len, ,bs, hidden_dim*num_directions] = [1, bs, hidden_dim]
        # hidden = [num_layers*num_directions, bs, hidden_dim] = [1, bs, hidden_dim]
        
        prediction = self.linear(outputs.squeeze(0))
        
        # [bs, decoder_vocab_dim]
        
        return prediction, hidden
    
    
    
class Seq2SeqBase(nn.Module):
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src_frames, target, teacher_forcing_ratio = 0.5):
        
        # src_frames [bs, src_len] = [bs, 32, 3, 224, 224]
        # target = [target_len, bs] = [20, bs]
        
        target = target.permute(1,0)
        batch_size = target.shape[1]
        target_len = target.shape[0] # although this has been hardcoded to 20 as of now
        
        target_vocab_size = self.decoder.output_vocab_dim
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        
        # [trg_len, bs, vocab_dim]
        
        hidden, _ = self.encoder(src_frames)
        
        # hidden [1, bs, hidden_dim]
        
        inp = target[0,:]
        
        # first input to the token is <sos> token. TODO add this in the sentences. 
        
        for i in range(1, target_len):
            
            decoder_output, hidden = self.decoder(inp, hidden)
            
            outputs[i] = decoder_output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            max_prob_word = decoder_output.argmax(1)
            
            inp = target[i] if teacher_force else max_prob_word
        
        return outputs[1:]
    