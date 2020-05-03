import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k, TranslationDataset
from torchtext.data import Field, BucketIterator

import spacy
import math, random, time , numpy as np



class EncoderLayer(nn.Module):
    
    def __init__(self, hid_dim, posfwd_dim, num_heads, dropout, device):
        
        super().__init__()
        
        self.device = device
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        
        self.powsfwd_layer_norm = nn.LayerNorm(hid_dim)
        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, num_heads, dropout, device)
        
        self.positionwise_feedforward = PositionWiseFeedForwardLayer(hid_dim, posfwd_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        # src = [bs, src_len]
        # src_mask = [bs, src_len] 
        
        _src, _ = self.self_attention(src, src, src, src_mask)
        # _src = [bs, query_len, hid_dim] or [bs, src_len, hid_dim]
        
        src = self.self_attn_layer_norm(self.dropout(_src) + src)
        # src = [bs, src_len, hid_dim]
        
        _src = self.positionwise_feedforward(src)
        # _src = [bs, src_len, hid_dim]
        
        src = self.posfwd_layer_norm(self.dropout(_src) + src)
        
        # src = [bs, src_len, hid_dim]
        
        return src
        
       

    
class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, hid_dim, num_heads, dropout, device):
        
        super().__init__()
        self.device = device
        
        self.num_heads = num_heads
        self.hid_dim = hid_dim
        
        self.head_dim = hid_dim // num_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
       
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask=None):
        
        batch_size = query.shape[0]
        
        # query = [bs, query_len, hid_dim]
        # key = [bs, key_len, hid_dim]
        # value = [bs, value_len, hid_dim]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # Q = [bs, query_len, hid_dim]
        # K = [bs, key_len, hid_dim]
        # V = [bs, value_len, hid_dim]
        
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        # Q = [bs, num_heads, query_len, head_dim]
        # K = [bs, num_heads, key_len, head_dim]
        # V = [bs, num_heads, value_len, head_dim]
        
        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        
        # energy = [bs, num_heads, query_len, key_len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        # In the energy tensor, if value is 0, replace it with -1e10.
        
        attention = torch.softmax(energy, dim=-1)
        
        # attention = [bs, num_heads, query_len, key_len]
        
        x = torch.matmul(self.dropout(attention), V)
        
        # [bs, num_heads, query_len, key_len] * [bs, num_heads, value_len, head_dim] =>
        # x => [bs,num_heads,query_len,head_dim]
        # Here matrix multiplication is possible as we know that key_len and 
        # value_len have same dimensions.
        
        x = x.permute(0,2,1,3).contiguous()
        
        # x = [bs, query_len, num_heads, head_dim]
        # contiguous() creates a copy of tensor so the order of elements would be 
        # same as if tensor of same shape created from scratch.

        x = x.view(batch_size, -1, self.hid_dim)
        
        # x = [bs, query_len, hid_dim]
        
        x = self.fc_o(x)
        
        # x = [bs, query_len, hid_dim]
        
        # attention = [bs, num_heads, query_len, key_len]
        
        return x, attention
        

        
class PositionWiseFeedForwardLayer(nn.Module):

    
    def __init__(self, hid_dim, posfwd_dim, dropout):
        
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, posfwd_dim)
        
        self.fc_2 = nn.Linear(posfwd_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        # x = [bs, seq_len, hid_dim] seq_len because it can be src_len or trg_len
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [bs, seq_len, posfwd_dim]
        
        x = self.fc_2(x)
        # x = [bs, seq_len, hid_dim]
        
        return x
    
    
    
    
    
class DecoderLayer(nn.Module):
    
    def __init__(self, hid_dim, num_heads, posfwd_dim, dropout, device):
        
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, num_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, num_heads, dropout, device)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.posfwd_layer_norm = nn.LayerNorm(hid_dim)
        self.posfwd_layer = PositionWiseFeedForwardLayer(hid_dim, posfwd_dim, dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        # trg = [bs, trg_len, hid_dim]
        # enc_src = [bs, src_len, hid_dim]
        # trg_mask = []
        # src_mask = []
        
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # _trg = [bs, trg_len, hid_dim]
        
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [bs, trg_len, hid_dim]
        
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # _trg = [bs, src_len, hid_dim]
        # attention = [bs, num_heads, trg_len, src_len] 
        # ([bs, num_heads, query_len, key_len]=>output format)
        # this is because here the query is the target sentence and key and values are
        # from source sentence. 
        
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [bs, trg_len, hid_dim]
        
        _trg = self.posfwd_layer(trg)
        
        trg = self.posfwd_layer_norm(trg + self.dropout(_trg))
        
        # trg = [bs, trg_len, hid_dim]
        
        return trg, attention
        
            