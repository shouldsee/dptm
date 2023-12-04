
 
# from copyreg import pickle

import pickle
import os
os.environ['KERAS_BACKEND']='torch'
 
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
 
 
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
# from keras.utils.np_utils import to_categorical
# from keras.layers import Dense, Input, Flatten
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
# from keras.models import Model

# categories = ['alt.atheism', 'soc.religion.christian']


from torch.utils.data import DataLoader, Dataset
from dptm.examples.util import TextSamplerDataset
from dptm.examples.util import cycle
import torch

class DatasetNews20(object):
    def __init__(self,path=None):

        categories = None
        if path is None:
            path =  __file__+'.dat'

        MAX_SEQUENCE_LENGTH = 1000
        # MAX_NB_WORDS        = 20000
        MAX_NB_WORDS        = 5000
        self.V = MAX_NB_WORDS

        fn = f'{path}/data.pkl'
        if not os.path.exists(fn):            
            # newsgroups_train = fetch_20newsgroups(subset=['all'], shuffle=True, 
            # newsgroups_train = fetch_20newsgroups(data_home = __file__+'.dat', subset='all', shuffle=True, 
            newsgroups_train = fetch_20newsgroups(data_home = path, subset='train', shuffle=True, 
                                                categories=categories,)

                                            

            print (newsgroups_train.target_names)
            print (len(newsgroups_train.data))
            print("\n".join(newsgroups_train.data[0].split("\n")[10:15]))

            texts = []

            labels = newsgroups_train.target
            texts  = newsgroups_train.data





            tokenizer = Tokenizer(num_words=MAX_NB_WORDS,oov_token='<oov>')
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)


            spidx = len(sequences)//5*4  
            corp_train = sequences[:spidx]
            corp_test  = sequences[spidx:]


            print('[concating]')
            def _concat(sent_list):
                size = sum(map(len,sent_list))
                xarr = np.zeros(size, dtype='int')
                i = 0
                for v in sent_list:
                    xarr[i:i+len(v)] = v
                    i = i+len(v)
                return xarr

            data_train = _concat(corp_train)
            data_val  = _concat(corp_test)
            data_train = torch.tensor(data_train)
            data_val = torch.tensor(data_val)
            # data_train,data_val = 
            # data_train = sum(corp_train,[])
            print('[concating_done]')
            # print('[concating_done]')
            # [0].join(corp_train)
            # sum([],)

            print (sequences[0][:10])

            print([len(x) for x in sequences[:10]])
            print([len(x) for x in texts[:10]])

            ### long-range interaction?
            ### recursive structure of the 
            print(sum([len(x) for x in sequences[:]]))
            with open(fn,'wb') as f:
                # f.write(pick)
                # pickle
                pickle.dump((data_train,data_val,tokenizer), f)

        else:
            with open(fn,'rb') as f:
                (data_train,data_val,tokenizer) = pickle.load(f)
        self.tokenizer = tokenizer
        self.V = tokenizer.num_words


        self.train_dataset = TextSamplerDataset(data_train, SEQ_LEN, NUM_SEGMENTS)
        self.val_dataset   = TextSamplerDataset(data_val, SEQ_LEN, NUM_SEGMENTS)
        self.train_loader  = cycle(DataLoader(self.train_dataset, batch_size = BATCH_SIZE))
        self.val_loader    = cycle(DataLoader(self.val_dataset, batch_size = BATCH_SIZE))
        # print(self.tokenizer.word_index)
        # breakpoint()
    def decode_tokens(self,tokens):
        tokens = tokens[None]
        tokens = tokens.detach().cpu().numpy().tolist()
        tokens = [list(x) for x in tokens]
        # print(self.tokenizer.word_index.get(tokens[0,0]))
        return self.tokenizer.sequences_to_texts(tokens)[0]
        # return self.tokenizer.sequences_to_texts(tokens)
        # [0]





####
if 0:
    ### core logic of elbo

    ## inference
    def get_logp():
        lp_external = None
        emb_wke     = None

        ### decide whether the token is emittable. (model selection)

        ### (B, L, E)
        
        ### (B, H, D)
        ### (1, H, D, H)


        ### (B, H2)
        ### (B, H3)

        pass
    # layer1 = 


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



# from examples.enwik8_simple import DatasetEnwiki8
if __name__ == '__main__':

    dat = DatasetEnwiki8()
    assert 0

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
                vdata = next(dat.val_loader)
                # if not len(val_loader):
                loss = loss_fn(vdata)
                print(f'validation loss: {loss.item():.4f}')

        optim.zero_grad()

        model.train()
        grad_accum_every = BATCH_SIZE / MAX_BATCH_SIZE
        # breakpoint()
        xdata = next(dat.train_loader)
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
