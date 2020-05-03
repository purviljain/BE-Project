from preprocess_attention import read_json, create_caption_dict, create_video_objects, clean_caption, gather_text, get_transforms
from preprocess_attention import Video, VideoDataset, create_vocab, get_caption_dict
from base import EncoderBase, DecoderBase, Seq2SeqBase
from torch.utils.data import DataLoader, Dataset
import torch
import random
from preprocess import *
from base import *
import time
from gru_summ_model import SummarizationModel

data = read_json()

captions_dict = get_caption_dict()

caption_text = gather_text(captions_dict)

vocab, vocab_to_int, int_to_vocab = create_vocab(caption_text)

train_video_objects = create_video_objects(0, 1024, data)

transform = get_transforms()

train_data = VideoDataset(train_video_objects, vocab_to_int, transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=False)

#valid_video_objects = create_video_objects(10, 14, data)

#valid_data = VideoDataset(valid_video_objects, vocab_to_int, transform)

#valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False)

INPUT_DIM = CNN_FEATURE_DIM = 1024
OUTPUT_DIM = len(vocab) #len(vocab) #import vocab here
ENCODER_HIDDEN_DIM = 384
DECODER_HIDDEN_DIM = 384
EMBEDDING_DIM = 100
DROPOUT = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attention = AttentionLayer(ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM)
encoder = EncoderAttention(CNN_FEATURE_DIM, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM, DROPOUT)
decoder = DecoderAttention(OUTPUT_DIM, EMBEDDING_DIM, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM, DROPOUT, attention)
# summ_model = SummarizationModel(512, 1024)
# summ_model.load_state_dict(torch.load('sum_model_epoch100_loss_0.004713772732999091.pt'))
# summ_model.to(device)

model = Seq2SeqAttention(encoder, decoder, device)
model = nn.DataParallel(model, devices=[2,3]).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
        
model.apply(init_weights)


# model.module.encoder.gru.weight_hh_l0 = summ_model.gru.weight_hh_l0
# model.module.encoder.gru.weight_ih_l0 = summ_model.gru.weight_ih_l0
# model.module.encoder.gru.bias_hh_l0 = summ_model.gru.bias_hh_l0
# model.module.encoder.gru.bias_ih_l0 = summ_model.gru.bias_ih_l0
# model.module.encoder.mlp1.weight = summ_model.mlp1.weight
# model.module.encoder.mlp1.bias = summ_model.mlp1.bias
# model.module.encoder.mlp2.weight = summ_model.mlp2.weight
# model.module.encoder.mlp2.bias = summ_model.mlp2.bias

# model.module.encoder.mlp1.weight.requires_grad = False
# model.module.encoder.mlp1.bias.requires_grad = False 
# model.module.encoder.mlp2.weight.requires_grad = False
# model.module.encoder.mlp2.bias.requires_grad = False

def get_opt_loss():
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    return optimizer, criterion


def count_parameters(model):
    parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return parameters_count


def train(model, loader, opt, criterion, clip):
    
    model.train()
    epoch_loss = 0.0
    
    for frames, captions in loader:
        
        # frames = [bs, seq_len, 3, 224, 224] = [bs, 32, 3, 224, 224]
        # captions = [bs, num_captions, seq_len] = [bs, 20, 20]
        
        #captions = captions.permute(1, 2, 0).type(torch.LongTensor)
         # captions = [num_captions, seq_len, bs] = [20, 20, bs]
        
        frames = frames.to(device)
        captions = captions.to(device)
       
        
        for i in range(captions.shape[1]):
            
            opt.zero_grad()
            
            output = model(frames, captions[:,i,:].long())
            
            # output [trg_len, bs, output_vocab_dim]
            # captions[i] = [trg_len, bs]
            
            output_dim = output.shape[-1]
            
            output = output.view(-1, output_dim)
            # output = [(trg_len-1)*bs, output_dim]
            
            target = captions[:,i,:].T[1:].reshape(-1)
            # target = [(trg_len-1)*bs]
            
            loss = criterion(output, target.long())
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
            opt.step()
        
            epoch_loss += loss.item()
        
    
    #print(epoch_loss/len(loader))
    return epoch_loss / (len(loader)*20)


def evaluate(model, loader, criterion):
    
    model.train()
    epoch_loss = 0.0
    
    for frames, captions in loader:
        
        # frames = [bs, seq_len, 3, 224, 224] = [bs, 32, 3, 224, 224]
        # captions = [bs, num_captions, seq_len] = [bs, 20, 20]
        
        captions = captions.permute(1, 2, 0).type(torch.LongTensor)
         # captions = [num_captions, seq_len, bs] = [20, 20, bs]
        
        frames = frames.to(device)
        captions = captions.to(device)
       
        with torch.no_grad():
            for i in range(captions.shape[0]):

               

                output = model(frames, captions[i])

                # output [trg_len, bs, output_vocab_dim]
                # captions[i] = [trg_len, bs]

                output_dim = output.shape[-1]

                output = output[1:].view(-1, output_dim)
                # output = [(trg_len-1)*bs, output_dim]

                target = captions[i][1:].view(-1)
                # target = [(trg_len-1)*bs]

                loss = criterion(output, target)
                
                epoch_loss += loss.item()


    #print(epoch_loss/len(loader))
    return epoch_loss / len(loader)
 
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 2
CLIP = 1
optimizer, criterion = get_opt_loss()

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    #valid_loss = evaluate(model, valid_loader, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    torch.save(model.state_dict(), 'attn{0}_{1}.pt'.format(epoch,round(train_loss,3)))

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
  



