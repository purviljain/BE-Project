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
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from string import punctuation

with open('./videodatainfo_2017_ustc.json', 'r') as f:
    parsed = json.load(f)

data = pd.DataFrame(parsed['sentences'])
# data.head()

with open('vocab_to_int.pickle', 'rb') as handle:
    vocab_to_int = pickle.load(handle)

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

    
    def calculate_output_captions(self):
        
        self.output_captions = []
        
        for ids in self.padded_captions:
            ids = ids[1:]
            ids.append(0)
            
            self.output_captions.append(ids)
        
        return self.output_captions
    
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
        
    
    # the following function is not being used currently.
    
    def get_vgg_tensor(self, transforms, model):
    
        vgg_features = []
        path = './frames/'+self.video_id+'/'
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
            self.vgg_tensor = model(feats)

        del feats
        gc.collect()
        model = model.cpu()
        self.vgg_tensor = vgg_tensor.cpu()

        return self.vgg_tensor

    
    def __str__(self):
        return '{}'.format(self.video_id)
    
    
def get_transforms():
    transform = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
    #vgg = models.vgg16(pretrained=True)
    #vgg.classifier = nn.Sequential(*[vgg.classifier[i] for i in range(4)])
    
    return transform

transform = get_transforms()    
       
def clean_caption(captions:list):
    
    clean_captions = []
    for caption in captions:
        caption = ''.join([ch for ch in caption if ch not in punctuation])
        caption  = '<sos> ' + caption + ' <eos>'
        clean_captions.append(caption)
    
    return clean_captions



def create_video_objects(num_objects)->list:
    '''Creates video objects with all the properties and methods.'''
    
    video_objects = [None] * num_objects
    for i in range(len(video_objects)):

        video_id = 'video'+str(i)
        captions = list(data[data['video_id'] == video_id].caption)
        captions = clean_caption(captions)
        sentence_ids = list(data[data['video_id'] == video_id].sen_id)

        video_objects[i] = Video(video_id=video_id, captions=captions,sentence_ids=sentence_ids)
    
    return video_objects
    
    
    
    