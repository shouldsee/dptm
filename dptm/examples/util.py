
from torch.utils.data import DataLoader, Dataset
import torch

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

