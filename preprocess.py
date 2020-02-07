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
from collections import Counter
from string import punctuation


with open('./videodatainfo_2017_ustc.json', 'r') as f:
    parsed = json.load(f)
    
    
print(parsed.keys())
data = pd.DataFrame(parsed['sentences'])
print(data.head())

def create_caption_dict()->tuple:
    '''
    Creates two dictionaries: 
    1.) caption_dict with key as video_id and value as list of captions for that video.
    {video_id:[caption_1, caption_2, caption_3 ....], ..}   
    2.) sentence_ids which has video_id as the key and list of sentence_ids as value. 
    sen_ids are ids attached with each caption.
    '''
    captions_dict = {}
    sentence_ids = {} 

    for video_id in list(data['video_id'].unique()):

        caption_list = list(data[data['video_id'] == video_id].caption)
        sentence_list = list(data[data['video_id'] == video_id].sen_id)
        captions_dict[video_id] = caption_list
        sentence_ids[video_id] = sentence_list
        
    return captions_dict, sentence_ids



# this will take time    
captions_dict = create_caption_dict()[0] 

with open('captions_dict.pickle', 'wb') as handle:
    pickle.dump(captions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('captions_dict.pickle', 'rb') as handle:
    captions_dict = pickle.load(handle)


def gather_text(captions_dict:dict)->list:
    '''Returns all the captions from all the videos as a list'''
    
    caption_text = []
    for k, v in captions_dict.items():
        for caption in v:
            caption = ''.join([ch for ch in caption if ch not in punctuation])
            caption  = '<sos> ' + caption + ' <eos>'
            caption_text.append(caption)
    
    return caption_text


caption_text = gather_text(captions_dict)

# test function
print(caption_text[0:10])

def create_vocab(caption_text:list)->tuple:
    '''
    Creates a vocabulary from the text dataset. Returns vocab_to_int which maps each word
    to an integer and vice-versa.
    '''
    words = []
    for text in caption_text:
        for word in text.split():
            words.append(word)
    
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    #vocab['<sos>'], vocab['<eos>'] = -1, -2
    vocab_to_int = {word:i for i, word in enumerate(vocab, 1)}
    int_to_vocab = {v:k for k,v in vocab_to_int.items()}
    
    return vocab, vocab_to_int, int_to_vocab


vocab, vocab_to_int, int_to_vocab = create_vocab(caption_text)

with open('vocab_to_int.pickle', 'wb') as handle:
    pickle.dump(vocab_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)


    
# The following functions have been wrapped up into the video class
# for modularity and are here for reference only.


def caption_to_int(caption_text:list, vocab_to_int:dict)->list:
    '''
    Numericalizes the captions by referring to the mapping dictionary.
    '''
    captions_to_int = []
    for i, caption in enumerate(caption_text):
        ids = []
        words = caption.split()

        for word in words:

            ids.append(vocab_to_int[word])

        captions_to_int.append(ids)
    
    return captions_to_int  
    
seq_len = 20

def pad_sequences()->list:
    '''
    Pads sequences/captions with 0's if their length is less than seq_len else truncates
    them to seq_len.
    '''
    padded_captions = []
    for ids in captions_to_int:
        ids = np.array(ids)
        if len(ids) > 20:
            ids = ids[:20]
        else:
            ids = np.pad(ids,(seq_len-len(ids),0),'constant')

        padded_captions.append(list(ids))
    
    return padded_captions

def get_vgg_tensor(video_id, transforms, model):
    
    vgg_features = []
    path = './frames/'+video_id+'/'
    frames = os.listdir(path)
    for frame in frames:
        img = Image.open(path+frame)
        img = transforms(img)
        
        #features = model(img.unsqueeze(0))
        
        vgg_features.append(img)
    
    feats = torch.stack(vgg_features, dim=0)
    feats = feats.cuda()
    model = model.cuda()
    
    with torch.no_grad():
        vgg_tensor = model(feats)
    
    del feats
    gc.collect()
    model = model.cpu()
    vgg_tensor = vgg_tensor.cpu()

    return vgg_tensor
        
