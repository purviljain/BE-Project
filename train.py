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
from video import create_video_objects, Video, get_transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from string import punctuation


with open('vocab_to_int.pickle', 'rb') as handle:
    vocab_to_int = pickle.load(handle)

transform = get_transforms()    
    
# Memory error on 16GB RAM for 1024 objects, :(
video_objects = create video_objects(64)

class VideoDataset(Dataset):
    
    def __init__(self, video_objects, vocab_to_int, transform):
        self.video_objects = video_objects
        self.call_funcs(vocab_to_int, transform)
        
    def call_funcs(self, vocab_to_int, transform):
        
        for video in self.video_objects:
            video.captions_to_token(vocab_to_int)
            video.pad_captions(20)
            video.get_video_frames(transform)
            
    def __len__(self):
        return len(video_objects)
        
    def __getitem__(self, index):
        return video_objects[index].frames, torch.tensor(video_objects[index].padded_captions) 
    
    

video_data = VideoDataset(video_objects, vocab_to_int, transform)    
loader = DataLoader(video_data, batch_size=16, shuffle=False)

class Encoder(nn.Module):
    
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        
        self.vgg_model = self.get_vgg()
        
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim)    
    
    def get_vgg(self):
        
        vgg = models.vgg16(pretrained=True)
        vgg.classifier = nn.Sequential(*[vgg.classifier[i] for i in range(4)])
        
        return vgg
    
    def forward(self, x):
        
        # x = [bs, seq_len, 3, 224, 224]
        
        with torch.no_grad():
            vgg_features = self.vgg_model(x)
        
        # vgg_features = [seq_len, bs, 4096]
        
        outputs, hidden = self.gru(vgg_features)
        
        # outputs = [seq_len, bs, hidden_dim*num_directions] = [32, 32, 512]
        # hidden = [num_layers*num_directions, bs, hidden_dim] = [1, 32, 512]
        
        return hidden
    
        
class Decoder(nn.Module):
    
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
        
        # outputs = [seq_len, bs, hidden_dim*num_directions] = [1, bs, hidden_dim]
        # hidden = [num_layers*num_directions, bs, hidden_dim] = [1, bs, hidden_dim]
        
        prediction = self.linear(outputs.squeeze(0))
        
        # [bs, decoder_vocab_dim]
        
        return prediction, hidden
    

class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src_frames, target, teacher_forcing_ratio = 0.5):
        
        # src_frames [src_len, bs] = [32, bs]
        # target = [target_len, bs] = [20, bs]
        
        batch_size = target.shape[1]
        target_len = target.shape[0] # although this has been hardcoded to 20 as of now
        
        target_vocab_size = self.decoder.output_vocab_dim
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        
        # [trg_len, bs, vocab_dim]
        
        hidden = self.encoder(src_frames)
        
        # hidden [1, bs, hidden_dim]
        
        inp = target[0,:]
        
        # first input to the token is <sos> token. TODO add this in the sentences. 
        
        for i in range(1, target_len):
            
            decoder_output, hidden = self.decoder(inp)
            
            outputs[i] = decoder_output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            max_prob_word = decoder_output.argmax(1)
            
            inp = target[i] if teacher_force else max_prob_word
        
        return outputs
    
    
input_dim = 4096
output_vocab_dim = len(vocab_to_int.keys()) #len(vocab) #import vocab here
encoder_hid_dim = 512
decoder_hid_dim = 512
embedding_dim = 256
dropout = 0.4

encoder = Encoder(hidden_dim=encoder_hid_dim, input_dim=input_dim)
decoder = Decoder(embedding_dim, decoder_hid_dim, output_vocab_dim, dropout)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Seq2Seq(encoder, decoder, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)


# This function needs to be tested
def train(model, loader, opt, criterion, clip):
    
    model.train()
    epoch_loss = 0.
    
    for frames, captions in loader:
        
        # frames = [16, 32, 3, 224, 224]
        # captions = [16, 20, 20]
        frames = frames.permute(1, 0, 2, 3, 4)
        captions = captions.permute(1, 2, 0)
        
        for i in range(captions.shape[0]):
            
            opt.zero_grad()
            
            output = model(frames, captions[i])
            
            # output [trg_len, bs, output_vocab_dim]
            
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            
            target = captions[i][1:].view(-1)
            
            loss = criterion(output, target)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
            opt.step()
        
            epoch_loss += loss.item()
        
    
    return epoch_loss / len(loader)
            