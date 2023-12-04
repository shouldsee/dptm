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

import os
# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 50
MAX_BATCH_SIZE = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100

GENERATE_EVERY  = 500
PRIME_LENGTH    = 10
GENERATE_LENGTH = 25

SEQ_LEN = 50
# NUM_SEGMENTS = 4
NUM_SEGMENTS = 1

# helpers


from torch.utils.data import DataLoader, Dataset
from dptm.examples.util import TextSamplerDataset
from dptm.examples.util import cycle



class DatasetEnwiki8(object):
    DIR = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, path=f'{DIR}/data/enwik8.gz'):
        self.path = path
        with gzip.open(self.path) as file:
            X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
            trX, vaX = np.split(X, [int(90e6)])
            data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)
        self.V = 257
        self.train_dataset = TextSamplerDataset(data_train, SEQ_LEN, NUM_SEGMENTS)
        self.val_dataset   = TextSamplerDataset(data_val, SEQ_LEN, NUM_SEGMENTS)
        self.train_loader  = cycle(DataLoader(self.train_dataset, batch_size = BATCH_SIZE))
        self.val_loader    = cycle(DataLoader(self.val_dataset, batch_size = BATCH_SIZE))
    # def decode_tokens(self,tokens):
    #   pass

    def decode_token(self, token):
        return str(chr(max(32, token)))

    def decode_tokens(self, tokens):
        return ''.join(list(map(self.decode_token, tokens)))



class SimpleDBN(nn.Module):
  '''
  Dynamic probabilistic tree models
  '''
  def __init__(self, 
      params,
      device=None,
      ):
      super().__init__()

      self.params = params
      # is_ar_loss       = params.is_ar_loss
      # is_ar_loss = False

      V = params.V
      K = params.K
      E = params.E
      M = params.M
      T = params.T
      # self.params = AttrDict()
      if device is None:
        device = torch.device('cpu')
      self.device = device

      params.D = D =  3

      self.rnn = nn.GRU(E, E, D, batch_first=True,).to(device)

      self.emb_vocab  = nn.Parameter(nn.Linear(E, V,).weight.to(device))



  def sample(self, 
    tok_external, t, 
    is_ar_loss=False,
  ):        
    '''

    '''
    tok_external = tok_external.clone()    
    tok_next = tok_external
    h_next = None
    for i in range(t):
        output_logit, h_next = self.forward(tok_next, h_next)

        xp = output_logit[:,-1,:].exp().cumsum(dim=-1)
        idx = (xp > torch.rand_like(xp)).max(dim=-1)[1]
        tok_sampled = idx[:,None]
        # breakpoint()

        tok_external = torch.cat([tok_external,tok_sampled],dim=1)
        tok_next = tok_sampled
        h_next = h_next

    return tok_external



  def generate(self, 
    tok_external, t, 
    is_ar_loss=False,
  ):        
    '''
    single sample

    '''
    tok_external= tok_external[None]
    sampled = self.sample(tok_external, t)
    return sampled[0]


  def forward(self, tok_external, h0=None):
    tok_emb = self.emb_vocab[tok_external]
    B,L,E = tok_emb.shape
    if h0 is None:
        h0 = 0* tok_emb[:,0:1,:].expand(B,self.params.D,E).transpose(0,1) 

    output, hn = self.rnn(tok_emb, h0 )
    output_logit = output.matmul(self.emb_vocab.T).log_softmax(-1)
    return output_logit,hn



  def loss_joint_lp(self, tok_external):
    output_logit, hn = self.forward(tok_external, None)

    lp_joint_bm = torch.gather( output_logit, index=tok_external.unsqueeze(-1),dim=-1).squeeze()
    lp_joint_bm = lp_joint_bm.sum(dim=1)
    loss = - lp_joint_bm.sum(0)


    # loss = -( lp_joint_bm.logsumexp(-1).sum(0))
    return loss






def run_loop(model,dat, i, loss_fn,loss_fn_valid=None,
    NUM_BATCHES = 10000,
    lr=0.01,

):
    if loss_fn_valid is None:
        loss_fn_valid = loss_fn
    model.parameters()



    optim = torch.optim.RMSprop(model.parameters(), lr=lr)

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=1., desc=f'[training_epoch={i}]'):

        if i % VALIDATE_EVERY == 0:
            model.eval()
            # with torch.no_grad():
            if 1:
                vdata = []
                # dl = 
                lrep = 10
                for _ in range(lrep):
                    vdata += [next(dat.val_loader)]
                vdata= torch.cat(vdata,0)
                loss = loss_fn_valid(vdata)/lrep
                print(f'validation loss: {[round(x.item(),3) for x in loss]}')


                vdata = []
                # dl = 
                lrep = 3
                for _ in range(lrep):
                    vdata += [next(dat.train_loader)]
                vdata= torch.cat(vdata,0)
                loss = loss_fn_valid(vdata)/lrep
                print(f'TRAINING loss: {[round(x.item(),3) for x in loss]}')


        model.train()
        grad_accum_every = BATCH_SIZE / MAX_BATCH_SIZE
        # breakpoint()
        xdata = next(dat.train_loader)
        loss = loss_fn(xdata)
        (loss / grad_accum_every).backward()

        # if i% grad_accum_every == 0:
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.15)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.005)
        optim.step()
        optim.zero_grad()
        # optim.zero_grad()

        # for mlm_loss, aux_loss, is_last in model(next(train_loader), max_batch_size = MAX_BATCH_SIZE, return_loss = True):
        #     loss = mlm_loss + aux_loss
        #     (loss / grad_accum_every).backward()

        #     print(f'training loss: {mlm_loss.item():.4f} | aux_loss: {aux_loss.item():.4f}')

        #     if is_last:
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        #         optim.step()
        #         optim.zero_grad()


        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(dat.val_dataset)[:-1]
            inp = inp[:PRIME_LENGTH]
            prime = dat.decode_tokens(inp)

            sample = model.generate(inp, GENERATE_LENGTH)
            print(inp)
            print(sample)
            output_str = dat.decode_tokens(sample)

            print(f'%s %s' % (prime, '*' * 10))
            print(output_str)

from dptm.examples.util import decode_tokens


# from examples.enwik8_simple import DatasetEnwiki8
if __name__ == '__main__':
    from dptm.examples.news20 import DatasetNews20

    dat = DatasetEnwiki8()
    # dat = DatasetNews20()

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
    # from dptm.dbn import DBN
    from dptm.srbm import SRBM
    from dptm.crbm import CRBM

    L = (NUM_SEGMENTS * SEQ_LEN ) +1


    params = AttrDict(dict(
        L = L,
        # K = 5,

        K = 2,
        # K = 4,
        # K = 6,
        # K = 2,
        # K = 3,
        # K = 7,
        # H = 70,
        # Z = 30,

        # H = 50,
        # Z = 20,

        # H = 100,
        # Z = 2,

        # H = 170,
        # H = 30,
        # # Z = 3,
        # Z = 30,
        # H=10,
        # Z=10,


        H=100,
        Z=100,

        # H=4,
        # Z=4,


        # H = 3,
        E = 31,
        B = 3,
        M = 12,
        EPS = 1E-8,
        # is_ar_loss = True,

        # betasq           = 12. ,  ### less noise in posterior
        # betasq_2         = 12.  ,

        # betasq           = 5 ,  ### less noise in posterior
        # betasq           = 0.5 ,  ### less noise in posterior
        # betasq           = 2 ,  ### less noise in posterior
        betasq           = 0.5 ,  ### less noise in posterior
        betasq_2         = 5.  ,
        # betasq_2         = 10.  ,

        # inner_nsample = 17,  ### how many samples to be taken to estimate elbo

        # # ### mc
        # method = 'mc',
        # inner_nsample = 30,  ### how many samples to be taken to estimate elbo
        # inner_nstep   = 30,
        # inner_lr      = 0.15,

        # elbo
        method = 'elbo',
        inner_nsample = 35,  ### how many samples to be taken to estimate elbo
        # inner_nstep   = 10,
        # inner_nstep   = 25,
        # inner_nstep   = 5,
        # inner_nstep   = 25,
        inner_nstep   = 15,
        # inner_lr      = 0.50,
        # inner_lr      = 0.05,
        # inner_lr      = 0.15,
        # inner_lr      = 0.95,
        inner_lr      = 0.85,


        # inner_nstep   = 20,


        # inner_nstep   = 10,
        # inner_lr      = 0.25,
        # inner_lr      = 0.50,

    ))
    LEARNING_RATE = 0.0005
    # LEARNING_RATE = 0.005
    # LEARNING_RATE = 0.01
    params.V = dat.V
    # params.V = 256 + 1
    params.T = L


    # device = None  
    device = torch.device("cuda:0") 


    # model = DBN(params,device)
    model = SRBM(params,device)
    # model = CRBM(params,device)
    
    # loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=True)
    # loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=True)
    # loss_fn_valid  = lambda x: model.loss_joint_lp(x, is_ar_loss=1 )
    

    # # model = DPT(params,device)
    # model = GRUWrapper(params,device)

    for i in range(1000):

        if i>=1000:
            loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=1)
        else:
            loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=0)

        # loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=False)
        # loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=1)
        loss_fn_valid  = lambda x: model.loss_joint_lp(x, is_ar_loss=0 )
        loss_fn_valid  = lambda x: torch.stack([
            model.loss_joint_lp(x, is_ar_loss=0 ),
            model.loss_joint_lp(x, is_ar_loss=1 ),
            ] ,0)

        loss_fn  = lambda x: model.loss_joint_lp(x)
        loss_fn_valid  = lambda x: torch.stack([
            model.loss_joint_lp(x, ),
            model.loss_joint_lp(x, ),
            ] ,0)


        run_loop(model, dat, i, loss_fn, loss_fn_valid,lr=LEARNING_RATE)
    

