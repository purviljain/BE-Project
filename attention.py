import json, pickle, os, random
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
import torch.nn.functional as F


class EncoderAttention(nn.Module):
    
    def __init__(self, cnn_feature_dim, encoder_hidden_dim, decoder_hidden_dim, dropout, device):
        
        super().__init__()
        
        self.device = device

        self.gru = nn.GRU(input_size=cnn_feature_dim, hidden_size=encoder_hidden_dim, bidirectional=True)
        
        self.linear = nn.Linear(in_features=encoder_hidden_dim*2, out_features=decoder_hidden_dim)
        
        self.tanh = nn.Tanh()
       
        self.dropout = nn.Dropout(dropout)
        
        #self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        
        #self.mlp1 = nn.Linear(hidden_dim, 1)
        
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
        
    
    def forward(self, src):
        
        # src = [bs, src_len, 3, 224, 224]
        
        src = src.permute(1,0,2,3,4)
        # src = [src_len, bs, 3, 224, 224]
        
        lenet_features = self.get_cnn_features(src)
        # lenet_features = [src_len, bs, 1024]
        
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(lenet_features)
        
        # outputs = [src_len, bs, enc_hid_dim*2]
        # hidden = [num_layers*num_directions, bs, enc_hid_dim] = [2, bs, enc_hid_dim]
        
        # The encoder is bidirectional and hence gives 2 context vectors - one forward and one backward.
        # However, the decoder is unidirectional and hence would accept only one context vector. 
        # Hence, we need to combine the context vectors which are actually the last 2 elements of the 
        # hidden vector. hidden is of the form - [forward_1, backward_1, forward_2, backward_2, ... fwd_l, bwd_l]
        # where l is the number of layers in the encoder. So, the top-layer's fwd and bwd states are 
        # hidden[-2] and hidden[-1]. We concat these two vectors, pass it through a linear layer and a tanh
        # activation.
        
        context = self.tanh(self.linear(torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)))
        
        #summary_scores = self.mlp2(self.mlp1(outputs))
        #summary_scores = summary_scores.view(-1,1).T
        
        # context = [bs, dec_hid_dim]
        return outputs, context  
    
    
class AttentionLayer(nn.Module):
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        
        super().__init__()
        
        self.energy_linear = nn.Linear(in_features=encoder_hidden_dim*2 + decoder_hidden_dim, 
                                       out_features=decoder_hidden_dim)
        
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        
        # decoder_hidden = [bs, dec_hid_dim]
        # encoder_outputs = [src_len, bs, enc_hid_dim*2]
        
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        
        decoder_hidden = decoder_hidden.unsqueeze(1)
        decoder_hidden = decoder_hidden.repeat(1, src_len, 1)
        
        # decoder_hidden = [bs, src_len, dec_hid_dim]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # encoder_outputs = [bs, src_len, enc_hid_dim*2]
        
        energy = torch.tanh(self.energy_linear(torch.cat([decoder_hidden, encoder_outputs], dim=2)))
        
        # energy = [bs, src_len, dec_hid_dim]
        
        attention = self.v(energy).squeeze(2)
        # attention = [bs, src_len]
        
        a = F.softmax(attention, dim=1)
        
        return a
    
    
class DecoderAttention(nn.Module):
    
    def __init__(self, output_vocab_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout, attention):
        
        super().__init__()
        
        self.output_vocab_dim = output_vocab_dim
        
        self.attention = attention
        
        self.embedding = self.load_glove()
        
        self.gru = nn.GRU(input_size=(encoder_hidden_dim*2)+embedding_dim, hidden_size=decoder_hidden_dim)
        
        self.linear = nn.Linear((encoder_hidden_dim*2)+embedding_dim+decoder_hidden_dim, output_vocab_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def load_glove(self):


        weights_matrix = np.load('glove.npy')
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix),freeze=True)

        return embedding


    def forward(self, inp, hidden, encoder_outputs):
        
        # inp = [bs] as decoder produces only one word per time-step
        # hidden = [bs, dec_hid_dim]
        # encoder_outputs = [src_len, bs, enc_hid_dim*2]
        
        inp = inp.unsqueeze(0)
        # inp [1, bs], introduce the seq_len dimension which is 1 since 
        # only word is produced by the decoder in one fwd pass
        
        embed = self.dropout(self.embedding(inp)) 
        # embed = [1, bs, emb_dim]
        
        a = self.attention(hidden, encoder_outputs)
        # a = [bs, src_len]
        
        a = a.unsqueeze(1)
        # a = [bs, 1, src_len]
        
        encoder_outputs = encoder_outputs.permute(1,0,2)
        # encoder_outputs = [bs, src_len, enc_hid_dim*2]
        
        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [bs, 1, enc_hid_dim*2]
        
        weighted = weighted.permute(1,0,2)
        # weighted = [1, bs, enc_hid_dim*2]
        
        gru_input = torch.cat((embed, weighted), dim=2)
        # gru_input = [1, bs, enc_hid_dim*2+emb_dim]
        
        self.gru.flatten_parameters()
        
        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        # output = [seq_len, bs, dec_hid_dim] = [1, bs, dec_hid_dim]
        # hidden = [1, bs, dec_hid_dim]
        
        embed = embed.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.linear(torch.cat((output, weighted, embed), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0)
    
    
class Seq2SeqAttention(nn.Module):
    
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
        
        encoder_outputs, hidden = self.encoder(src_frames)
        
        # hidden [1, bs, hidden_dim]
        
        inp = target[0,:]
        
        # first input to the token is <sos> token. TODO add this in the sentences. 
        
        for i in range(1, target_len):
            
            output, hidden = self.decoder(inp, hidden, encoder_outputs)
            
            outputs[i] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            max_prob_word = output.argmax(1)
            
            inp = target[i] if teacher_force else max_prob_word
        
        return outputs[1:]