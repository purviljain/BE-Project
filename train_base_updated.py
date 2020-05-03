from preprocess_base import read_json, create_caption_dict, create_video_objects, clean_caption, gather_text, get_transforms
from preprocess_base import Video, VideoDataset, create_vocab, get_caption_dict
from base import EncoderBase, DecoderBase, Seq2SeqBase
from torch.utils.data import DataLoader, Dataset
import torch
import random
from preprocess import *
from base import *
import time
from gru_summ_model import SummarizationModel


summ_model = SummarizationModel(512, 1024)
summ_model.load_state_dict(torch.load('sum_model_epoch100_loss_0.00471.pt'))
summ_model.to(device)
data = read_json()

captions_dict = get_caption_dict()

caption_text = gather_text(captions_dict)

vocab, vocab_to_int, int_to_vocab = create_vocab(caption_text)

train_video_objects = create_video_objects(0, 5, data)

transform = get_transforms()

train_data = VideoDataset(train_video_objects, vocab_to_int, transform)

train_loader = DataLoader(train_data, batch_size=1, shuffle=False)

valid_video_objects = create_video_objects(10, 14, data)

valid_data = VideoDataset(valid_video_objects, vocab_to_int, transform)

valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False)

INPUT_DIM = 1024
OUTPUT_DIM = len(vocab) #len(vocab) #import vocab here
ENCODER_HID_DIM = 512
DECODER_HID_DIM = 512
EMBEDDING_DIM = 256
DROPOUT = 0.4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


encoder = EncoderBase(hidden_dim=ENCODER_HID_DIM, input_dim=INPUT_DIM, device=device)
decoder = DecoderBase(EMBEDDING_DIM, DECODER_HID_DIM, OUTPUT_DIM, DROPOUT)
model = Seq2SeqBase(encoder, decoder, device)

model = nn.DataParallel(model).to(device)

pretrained_model = torch.load('base9_5.603.pt')
pretrained_model['module.encoder.mlp1.weight'] = summ_model.mlp1.weight
pretrained_model['module.encoder.mlp1.bias'] = summ_model.mlp1.bias
pretrained_model['module.encoder.mlp2.weight'] = summ_model.mlp2.weight
pretrained_model['module.encoder.mlp2.bias'] = summ_model.mlp2.bias

model.load_state_dict(pretrained_modelmodule.)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
#model.apply(init_weights)


# model.encoder.gru.weight_hh_l0 = summ_model.gru.weight_hh_l0
# model.encoder.gru.weight_ih_l0 = summ_model.gru.weight_ih_l0
# model.encoder.gru.bias_hh_l0 = summ_model.gru.bias_hh_l0
# model.encoder.gru.bias_ih_l0 = summ_model.gru.bias_ih_l0
# model.encoder.mlp1.weight = summ_model.mlp1.weight
# model.encoder.mlp1.bias = summ_model.mlp1.bias
# model.encoder.mlp2.weight = summ_model.mlp2.weight
# model.encoder.mlp2.bias = summ_model.mlp2.bias

model.module.encoder.mlp1.weight.requires_grad = False
model.module.encoder.mlp1.bias.requires_grad = False 
model.module.encoder.mlp2.weight.requires_grad = False
model.module.encoder.mlp2.bias.requires_grad = False



def get_opt_loss():
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
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
        
        captions = captions.permute(1, 2, 0).type(torch.LongTensor)
         # captions = [num_captions, seq_len, bs] = [20, 20, bs]
        
        frames = frames.to(device)
        captions = captions.to(device)
       
        
        for i in range(captions.shape[0]):
            
            opt.zero_grad()
            
            output = model(frames, captions[i])
            
            # output [trg_len, bs, output_vocab_dim]
            # captions[i] = [trg_len, bs]
            
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            # output = [(trg_len-1)*bs, output_dim]
            
            target = captions[i][1:].view(-1)
            # target = [(trg_len-1)*bs]
            
            loss = criterion(output, target)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
            opt.step()
        
            epoch_loss += loss.item()
        
    
    #print(epoch_loss/len(loader))
    return epoch_loss / len(loader)


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


N_EPOCHS = 8
CLIP = 1
optimizer, criterion = get_opt_loss()

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_loader, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    torch.save(model.state_dict(), 'base{0}_{1}.pt'.format(epoch,round(train_loss,3)))
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'best_model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')




