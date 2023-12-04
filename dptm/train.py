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



def run_loop(model,dat, epoch, loss_fn,loss_fn_valid=None,
    callback_generate=None,
    NUM_BATCHES = 10000,
    lr=0.01,

):
    if loss_fn_valid is None:
        loss_fn_valid = loss_fn
    model.parameters()



    optim = torch.optim.RMSprop(model.parameters(), lr=lr)
    # optim = torch.optim.SGD(model.parameters(), lr=lr)
    last_valid_loss = 0.
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=1., desc=f'[training_epoch={epoch}]'):

        if i % VALIDATE_EVERY == 0:
            model.eval()
            # with torch.no_grad():
            if 1:
                lrep = 1


                vdata = next(dat.val_loader)
                loss = loss_fn_valid(vdata)/lrep
                print(f'validation loss: {[round(x.item(),3) for x in loss]}')
                # if not isinstance(loss,(float,int,torch.Tensor)):
                # if 
                last_valid_loss = loss[0].item()
                vdata = next(dat.train_loader)
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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.005)
        optim.step()
        optim.zero_grad()
        # optim.zero_grad()



        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(dat.val_dataset)[:-1]
            if callback_generate is not None:
                callback_generate(model, epoch, i, inp, last_valid_loss)
            inp = inp[:PRIME_LENGTH]
            prime = dat.decode_tokens(inp)
            sample = model.generate(inp, GENERATE_LENGTH)
            print(inp[0].reshape(-1)[:20])
            # print(sample[0].reshape(-1)[:10])
            output_str = dat.decode_tokens(sample)

            print(f'%s %s' % (prime, '*' * 10))
            print(output_str)

from dptm.examples.util import decode_tokens

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from examples.enwik8_simple import DatasetEnwiki8
if __name__ == '__main__':
    # from dptm.examples.news20 import DatasetNews20 as DataCls
    from dptm.data.fashion_mnist import DataCls
    

    # device = None  
    device = torch.device("cuda:0") 
    dat = DataCls(device)



    from attrdict import AttrDict
    # from dptm.dbn import DBN
    from dptm.srbm import SRBM
    from dptm.crbm import CRBM

    L = (NUM_SEGMENTS * SEQ_LEN ) +1


    '''
    [training_epoch=0]:  97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▌   | 9698/10000 [01:28<00:02, 109.26it/s]validation loss: [-458.016, -458.776]
    TRAINING loss: [-439.118, -434.706]

    '''

    params = AttrDict(dict(
        L = L,
        K = 2,
        H=100,
        # Z=100,
        # E = 31,
        # E = 31    ,
        # E = 31  ,
        E = 21  ,
        # E = 231,
        B = 3,
        # M = 12,
        EPS = 1E-8,
        A = 10,

        betasq           = 0.5 ,  ### less noise in posterior
        betasq_2         = 5.  ,

        # M =60,
        T = 2,
        M = 20,   
        method = 'elbo',
        inner_nsample = 25,  ### how many samples to be taken to estimate elbo
        inner_nstep   = 3,
        inner_lr      = 0.2,


    ))
    LEARNING_RATE = 0.0005
    # LEARNING_RATE = 0.005
    # LEARNING_RATE = 0.00005
    params.V = dat.V
    params.T = L

    from dptm.zbml3 import ZBML3
    model = ZBML3(params,device)




    # model = DBN(params,device)
    # model = SRBM(params,device)
    # model = CRBM(params,device)
    
    # loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=True)
    # loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=True)
    # loss_fn_valid  = lambda x: model.loss_joint_lp(x, is_ar_loss=1 )
    

    # # model = DPT(params,device)
    from dptm.gru import GRUWrapper
    from dptm.gru import GRUWrapperGaussian
    # model = GRUWrapper(params,device)
    # from dptm.rbml1 import RBML1
    # model = RBML1(params,device)
    # from dptm.vael1 import VAEL1
    # model = VAEL1(params,device)

    # from dptm.vael2 import VAEL2
    # model = VAEL2(params,device)


    # from dptm.srbmgl2 import SRBMGL2
    # model = SRBMGL2(params,device)


    # from dptm.zbm import ZBM
    # model = ZBM(params,device)


    # from dptm.zbml2 import ZBML2
    # model = ZBML2(params,device)

    if not os.path.exists('out'): os.makedirs('out/') 

    alias = f'out/{model.__class__.__name__}'
    def callback_generate(model, epoch, i, inp,last_valid_loss,alias=alias, ):
        arr = model.sample((8,2,),inp)
        arr = arr.transpose(2,3)
        arr = torch.cat(list(torch.cat(list(arr),-2)),-1)
        fig = plt.figure()
        # plt.imshow( arr.detach().cpu().numpy(),origin='lower')
        im  = plt.imshow( arr.detach().cpu().numpy().T,origin='lower')
        ofn  = f"{alias}_e{epoch}_i{i}.png"
        plt.suptitle(f'ofn:{ofn}_loss:{last_valid_loss:.1f}')
        plt.colorbar(im)
        print(ofn)
        
        plt.savefig(ofn)
        plt.savefig(alias+'_last.png')
        plt.close(fig)






    for i in range(1000):

        if i>=1000:
            loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=1)
        else:
            loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=0)

        # loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=False)
        # loss_fn  = lambda x: model.loss_joint_lp(x,is_ar_loss=1)
        loss_fn_valid  = lambda x: model.loss_joint_lp(x, is_ar_loss=0 )
        loss_fn_valid  = lambda x: torch.stack([
            model.loss_joint_lp(x,  ),
            model.loss_joint_lp(x,  ),
            ] ,0)

        loss_fn  = lambda x: model.loss_joint_lp(x)
        loss_fn_valid  = lambda x: torch.stack([
            model.loss_joint_lp(x, ),
            model.loss_joint_lp(x, ),
            ] ,0)


        run_loop(model, dat, i, loss_fn, loss_fn_valid,lr=LEARNING_RATE,callback_generate=callback_generate)
    

