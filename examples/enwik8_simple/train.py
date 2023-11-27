'''
Adapted from https://github.com/lucidrains/compressive-transformer-pytorch

'''
# from compressive_transformer_pytorch import CompressiveTransformer
# from compressive_transformer_pytorch.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 10
MAX_BATCH_SIZE = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100

GENERATE_EVERY  = 500
PRIME_LENGTH    = 5
GENERATE_LENGTH = 1024

SEQ_LEN = 15
# NUM_SEGMENTS = 4
NUM_SEGMENTS = 1

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate model

# prepare enwik8 data
import os
DIR = os.path.dirname(os.path.realpath(__file__))

with gzip.open(f'{DIR}/data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, segments):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.segments = segments
        self.total_len = seq_len * segments

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.total_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.total_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.total_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN, NUM_SEGMENTS)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN, NUM_SEGMENTS)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))


# breakpoint()


# model = CompressiveTransformer(
#     num_tokens = 256,
#     dim = 512,
#     depth = 8,
#     seq_len = SEQ_LEN,
#     mem_len = SEQ_LEN,
#     cmem_len = SEQ_LEN // 4,
#     heads = 8,
#     memory_layers = [6,7,8]
# )

# model = AutoregressiveWrapper(model)
# model.cuda()

# # optimizer


from dptm.dptm import DPT

from attrdict import AttrDict

params = AttrDict(dict(
L = 5,
K = 7,
E = 21,
B = 3,
M = 12,
EPS = 1E-8,
# is_ar_loss = True,

betasq           = 12. ,  ### less noise in posterior
betasq_2         = 12.  ,

inner_nsample = 1,  ### how many samples to be taken to estimate elbo
inner_nstep   = 10,
inner_lr      = 0.03,

))
params.V = 256 + 1
params.T = (NUM_SEGMENTS * SEQ_LEN ) +1
# device = None  
device = torch.device("cuda:0") 
model = DPT(params,device)

loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=False)


optim = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)


NUM_BATCHES = 10

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=1., desc='training'):

    if i % VALIDATE_EVERY == 0:
        model.eval()
        # with torch.no_grad():
        if 1:
            vdata = next(val_loader)
            # if not len(val_loader):
            loss = loss_fn(vdata)
            print(f'validation loss: {loss.item():.4f}')

    optim.zero_grad()

    model.train()
    grad_accum_every = BATCH_SIZE / MAX_BATCH_SIZE
    # breakpoint()
    xdata = next(train_loader)
    loss = loss_fn(xdata)
    (loss / grad_accum_every).backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    # for mlm_loss, aux_loss, is_last in model(next(train_loader), max_batch_size = MAX_BATCH_SIZE, return_loss = True):
    #     loss = mlm_loss + aux_loss
    #     (loss / grad_accum_every).backward()

    #     print(f'training loss: {mlm_loss.item():.4f} | aux_loss: {aux_loss.item():.4f}')

    #     if is_last:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    #         optim.step()
    #         optim.zero_grad()


    # if i % GENERATE_EVERY == 0:
    #     model.eval()
    #     inp = random.choice(val_dataset)[:-1]
    #     inp = inp[:PRIME_LENGTH]
    #     prime = decode_tokens(inp)
    #     print(f'%s \n\n %s', (prime, '*' * 100))

    #     sample = model.generate(inp, GENERATE_LENGTH)
    #     output_str = decode_tokens(sample)
    #     print(output_str)
