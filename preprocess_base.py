import json, pickle, os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
import torch
import gc, sys, psutil
import numpy as np
from collections import Counter
from string import punctuation
from torch.utils.data import Dataset, DataLoader


def read_json():
    
    with open('./videodatainfo_2017_ustc.json', 'r') as f:
        parsed = json.load(f)

    data = pd.DataFrame(parsed['sentences'])
    
    return data

def create_caption_dict(data)->tuple:
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

def get_caption_dict():
    '''Returns captions_dict by reading from a pickle file.'''
    
    with open('captions_dict.pickle', 'rb') as handle:
        captions_dict = pickle.load(handle)
    
    return captions_dict

def gather_text(captions_dict:dict)->list:
    '''Returns all the captions from all the videos as a list'''
    
    caption_text = []
    for k, v in captions_dict.items():
        for caption in v:
            caption = ''.join([ch for ch in caption if ch not in punctuation])
            caption  = '<sos> ' + caption + ' <eos>'
            caption_text.append(caption)
    
    return caption_text


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

def get_transforms():
    '''Returns transforms to convert images from numpy arrays to tensors.'''
    
    transform = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
    #vgg = models.vgg16(pretrained=True)
    #vgg.classifier = nn.Sequential(*[vgg.classifier[i] for i in range(4)])
    
    return transform

class Video(object):
    
    def __init__(self, video_id, captions=[], sentence_ids=[]):
        self.video_id = video_id
        self.captions = captions
        self.sentence_ids = sentence_ids
        
    def captions_to_token(self, vocab_to_int):
        
        self.caption_tokens = []
        for caption in self.captions:
            ids = []
            words = caption.split()
            for word in words:
                    ids.append(vocab_to_int[word])
            self.caption_tokens.append(ids)
        
        #return self.caption_tokens
    
    
    def pad_captions(self, seq_len):
        
        self.padded_captions = []
        for ids in self.caption_tokens:
            ids = np.array(ids)
            if len(ids) > seq_len:
                ids = ids[:seq_len]
            else:
                ids = np.pad(ids,(0,seq_len-len(ids)),'constant')

            self.padded_captions.append(list(ids))
        
        #return self.padded_captions

  
    
    def get_video_frames(self, transform):
        
        frame_features = []
        path = './frames/'+self.video_id+'/'
        #frames = os.listdir(path)
        for frame in os.listdir(path):
            img = Image.open(path+frame)
            img = transform(img)
            frame_features.append(img)
        
        if len(frame_features) < 32:
            pad_val = 32 - len(frame_features)
            pad = torch.zeros(pad_val, 3, 224, 224)
            self.frames = torch.cat([torch.stack(frame_features, dim=0), pad], dim=0)
        else:
            self.frames = torch.stack(frame_features,dim=0)
        
        #return self.frames
  
    
    def __str__(self):
        return '{}'.format(self.video_id)
    
    

def clean_caption(captions:list):
    
    clean_captions = []
    for caption in captions:
        caption = ''.join([ch for ch in caption if ch not in punctuation])
        caption  = '<sos> ' + caption + ' <eos>'
        clean_captions.append(caption)
    
    return clean_captions
       

    
def create_video_objects(start_idx:int, end_idx:int,data)->list:
    '''Creates video objects with all the properties and methods.'''
    
    video_objects = [None] * (end_idx - start_idx)
    for i in range((end_idx - start_idx)):
        video_id = 'video'+str(start_idx + i)
        captions = list(data[data['video_id'] == video_id].caption)
        captions = clean_caption(captions)
        sentence_ids = list(data[data['video_id'] == video_id].sen_id)

        video_objects[i] = Video(video_id=video_id, captions=captions,sentence_ids=sentence_ids)
    
    return video_objects


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
        return len(self.video_objects)
        
    def __getitem__(self, index):
        return self.video_objects[index].frames, torch.tensor(self.video_objects[index].padded_captions) 
        
        
    
