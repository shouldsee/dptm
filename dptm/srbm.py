import math
import torch
import torch.nn as nn

# from transformer.data import data_utils
# from transformer.data import data_utils


import sys
DEBUG = '--debug' in sys.argv

from attrdict import AttrDict
# class Decoder(nn.Module):
PI = 3.1415926
def mc_update():
  ### 

  EPS = self.params.EPS


  node_z = node_z.requires_grad_(False)
  Z= params.Z
  NS = params.inner_nsample

  ### estimate elbo through sampling
  node_z_p = node_z
  node_z_noise = (torch.rand((B,Z,NS),device=device) < node_z_p.unsqueeze(-1)).float()

  node_h_p = (torch.einsum('bzn,hz->bnh', node_z_noise, whz) + bh[None,None,:]).sigmoid()
  node_h_noise = (torch.rand_like(node_h_p)<node_h_p).float()
  node_e_noise = torch.tensordot( node_h_noise, (w_hle), 1)

  x = torch.tensordot(node_e_noise[:, :,lidx ], emb_vocab_w.T,1).log_softmax(-1)
  # assert tok_external.max()+1<= params.V,(tok_external.max(),params.V)
  x = torch.gather(x, index=tok_external[:,None, lidx,None].expand((B,NS,len(lidx), 1)),dim=-1).squeeze(-1)
  lp_tok = x
  lp_b = 0
  lp_b += lp_tok.sum((2,))  ## (B,NS) 
  lp_b += torch.einsum('bnh,bnh->bn', node_h_noise, (node_h_p+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.
  lp_b += torch.einsum('bnh,bnh->bn', 1-node_h_noise, (1-node_h_p+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.

  lp_b += -torch.einsum('bzn,bz->bn', node_z_noise, (node_z_p+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.
  lp_b += -torch.einsum('bzn,bz->bn', 1-node_z_noise, (1-node_z_p+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.
  # lp_b += torch.einsum('bzn,bz->bn', node_z_noise, (node_z_p+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.
  # lp_b += torch.einsum('bzn,bz->bn', 1-node_z_noise, (1-node_z_p+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.

  ### approximate sample with log P(y,x,z)
  ## (B,NS)
  lp_b = (lp_b - torch.log(torch.tensor(NS,device=device)))
  lp_b_part = (lp_b.logsumexp(-1))
  
  p_bn = (lp_b - lp_b_part.unsqueeze(-1)).exp()

  ### divide partiaion by NS in log space
  lp_b = lp_b_part - torch.log(torch.tensor(NS,device=device))

  if DEBUG:
      pass
      # print(p_bn.sum().item())

      # breakpoint()
  # lp_b_p = lp_b.exp() 
  inner_lr 
  node_z  = ((1-inner_lr)*node_z.detach() + inner_lr*torch.einsum('bn,bzn->bz',p_bn.detach(),node_z_noise.detach()))
  # .detach()

  # node_h  = (1-inner_lr)*node_z + inner_lr*torch.einsum('bn,bzn->bz',p_bn,node_z_noise)
  del node_z_noise
  del node_h_noise
  del node_e_noise
  del node_h_p
  del node_z_p
  del lp_b_part
  del p_bn
  del lp_tok

  # torch.cuda.empty_cache()

  pass

class SRBM(nn.Module):
  '''
  A simple 3-layer RBM
  '''
  def __init__(self, 
      params,
      device=None,
      ):
      super().__init__()

      # self.d_model = d_model
      # self.dropout_emb  = nn.Dropout(dropout)
      self.params = params
      # is_ar_loss       = params.is_ar_loss
      # is_ar_loss = False

      V = params.V
      K = params.K
      E = params.E
      M = params.M
      T = params.T
      H = params.H
      Z = params.Z
      # self.params = AttrDict()
      if device is None:
        device = torch.device('cpu')
      self.device = device

      method = params.method


      L = params.L

      # self.emb_vocab  = nn.Embedding(V, E, padding_idx=data_utils.PAD, device=device)
      self.emb_vocab  = nn.Parameter(nn.Linear(E, V,).weight.to(device))
      #  (V, E)
      self.w_k        = nn.Parameter(nn.Linear(1,K*H*H).weight.reshape((K,H,H))[1:].to(device))
      self.b_k        = nn.Parameter(nn.Linear(1,(K-1)*H).weight.reshape((K-1,H)).to(device))
      self.w_hle      = nn.Parameter(nn.Linear(1,H*L*E).weight.reshape((H,L,E)).to(device))
      self.b_hle      = nn.Parameter(nn.Linear(1,L*E).weight.reshape((L,E)).to(device))
      self.whz      = nn.Parameter(nn.Linear(1,H*Z).weight.reshape((H,Z)).to(device))
      self.bh      = nn.Parameter(nn.Linear(1,H).weight.reshape((H,)).to(device))
      #  (2K, E,  E)


      ### precicision squared of gaussian
      # self.betasq     = torch.tensor(1.0).to(device)
      # self.betasq     = (1.0)

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



  def _inner_loop(self, bm_dict_next, t, max_t, is_ar_loss):
    '''
    ### t = 0 -> infer the first node, 1 obs + 0 hidden 1 choice
    ### t = 1 -> infer the 2nd node,   1 obs + 0 hidden        1 choice
    ### t = 2 -> infer the 3nd node.   2 obs + 1 hidden node   3 choice 2T-1

    ### beam search to update the node representation

    need to find the top m expansion

    node_e          (B,  M,  L,  E)   ### external nodes
    node_i_e        (B,  M,  2L,  E)   ### concat( external_nodes, internal nodes,dim=2)
    edge_children   (B,  M,  2L,  2)   
    edge_parent     (B,  M,  2L,  1)   
    edge_parent_k   (B,  M,  2L,  1)   

      (parent, left_child, right_child)
      choosing between (2T-3) edges to connect 
    
    ### copy the z's and do gradient steps.
    
    options         (B,  M,  2T-1, 2K)  (2L - 3)
    w_k                 (2, K, E,  E) ### only depends on parent and l/r
    w_k_ref             (E,  E,  K, 2) ### only
    w_k_comb            (2K, E,  E)    ### (parent_and_lr, parent, child)


    edge_c_e        (B,  M,  2L,  2,  E)
    edge_p_e        (B,  M,  2L,  1,  E)

    ### only (2L-3) options  are possible
    ### insert into one of the parent of the 2L nodes.
    ### expand the embedding to allow later propagation
      expand(2K, 2L) using stop grad?
    '''

    ### initialise from last state
    xbmd = bm_dict_next
    tok_external= xbmd.tok_external

    # node_h      = xbmd.node_h

    xd_next = AttrDict(xbmd.copy())
    # xd_next = AttrDict(
    #   tok_external= xbmd.tok_external,
    #   lp_ar = xbmd.lp_ar,
    # )

    params = self.params
    device = self.device

    is_mask_last     = is_ar_loss
    is_mask_last     = int(is_mask_last)

    w_k_comb         = self.w_k                            #  (2K, E,  E)
    emb_vocab_w      = self.emb_vocab                      ### (V, E)
    # betasq           = self.betasq        ### invert of sigma^2  

    betasq   = params.betasq
    betasq_2 = params.betasq_2        

    inner_nsample = params.inner_nsample
    inner_nstep   = params.inner_nstep
    inner_lr      = params.inner_lr


    K = params.K
    E = params.E
    M = params.M
    T = params.T
    B,L = tok_external.shape
    T = max_t
    # assert L<=T,(L,T)


    ### expand (2K*2L) options
    #node_h (B, K, H)

    H = params.H
    ### init the hidden
    ### K-1 H H
    w_k   = self.w_k
    b_k = self.b_k
    ### (H,E)
    w_hle = self.w_hle
    b_hle = self.b_hle

    
    
    ### update the internal representaion


    ### (B,L,E)
    tok_emb = self.emb_vocab[tok_external]
    node_z  = xbmd.node_z  ## (B,Z)
    node_h  = xbmd.node_h ##(B,H)
    whz = self.whz ### (H,Z) 
    bh  = self.bh ### (Z)

    EPS = params.EPS

    # method = 'elbo'
    # self.params.method  = method = 'mc'
    # self.params.method  = method = 'elbo'
    method = params.method
    
    lidx = range(0,t+1-is_mask_last)
    lidx = list(lidx)
    for i in range(inner_nstep):

        if method == 'elbo':

            ### (B,L,E)

            node_e = torch.tensordot( node_h, (w_hle), 1).sigmoid()

            x = torch.tensordot(node_e[:, lidx ], emb_vocab_w.T,1).log_softmax(-1)
            # assert tok_external.max()+1<= params.V,(tok_external.max(),params.V)
            x = torch.gather(x, index=tok_external[:, lidx, None],dim=-1).squeeze(-1)
            lp_tok = x

            ### (B,L)
            # node_e = torch.einsum(node_h, 3
            ## (B,L,H)

            dh_blh = torch.einsum( 'hle,ble->blh',w_hle[:,lidx], tok_emb[:,lidx]*(1-node_e[:,lidx])) 
            
            dh1 = torch.einsum('blh,bl->bh', dh_blh, ( 1- lp_tok.exp()))

            ### sigmoid(Wz + c), (B,H)
            h_p = (torch.einsum('bz,hz->bh', node_z, whz) + bh[None]).sigmoid()
            dh2 = h_p.log()


            lp_b = lp_tok.sum(1) + torch.einsum('bh,bh->b',node_h,h_p.log()) ### need to apply the entropy loss, because entropy is implicit on activation.
            lp_b += -torch.einsum('bz,bz->b', node_z, (node_z+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.
            lp_b += -torch.einsum('bz,bz->b', 1-node_z, (1-node_z+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.

            # lp_b = lp_b + torch.einsum('')
            ### update h
            p_h_update = inner_lr
            p_z_update = inner_lr


            def arr_random_update(arr,arr_next, p,):
                sel = (torch.rand_like(arr)<p)
                # out = (~sel) *arr + sel*arr_next
                out = (~sel) *arr + sel*(arr_next+arr)*0.5
                return out
            node_h = arr_random_update(node_h, (dh1+dh2).sigmoid(), p_h_update)    


            dz1 = torch.einsum('bh,hz->bhz',node_h, whz)
            dz1 = torch.einsum('bh,bhz->bz',(1 - h_p),dz1)


            ### update z
            # sel = (torch.rand_like(node_z)<p_z_update) 
            # node_z = dz1.sigmoid()

            node_z = arr_random_update(node_z, dz1.sigmoid(),p_z_update)
            node_h = node_h.detach()
            node_z = node_z.detach()
        elif method=='mc':
          assert 0

        lp_b;
        if DEBUG:
            print(f"[inner_i={i}]lp[shoud_incr]:{lp_b.sum().item():.2f}")
        # if DEBUG:
        #     print(f"[inner_i={i}]lp[shoud_incr]:{lp_b.sum().item():.2f}")
        #     # print(f'{node_z[0].detach().cpu().numpy()[:5]}')
        #     # x = (1-node_z_sig)*dz1
        #     # x =  -betasq * node_z + (1-node_z_sig)*dz1
            x = node_z
            print(f'{x[0].detach().cpu().numpy()[:5]}')
            # x = node_z.grad
            # print(f'{x[0].detach().cpu().numpy()[:5]}')

        # node_z[sel] = dz1[sel].sigmoid()
    EPS = self.params.EPS


    node_z
    Z= params.Z
    NS = params.inner_nsample

    ### estimate elbo through sampling
    node_z_p = node_z
    node_z_noise = (torch.rand((B,Z,NS),device=device) < node_z_p.unsqueeze(-1)).float()

    node_h_p = (torch.einsum('bzn,hz->bnh', node_z_noise, whz) + bh[None,None,:]).sigmoid()
    node_h_noise = (torch.rand_like(node_h_p)<node_h_p).float()
    node_e_noise = torch.tensordot( node_h_noise, (w_hle), 1).sigmoid() 

    x = torch.tensordot(node_e_noise[:, :,lidx ], emb_vocab_w.T,1).log_softmax(-1)
    # assert tok_external.max()+1<= params.V,(tok_external.max(),params.V)
    x = torch.gather(x, index=tok_external[:,None, lidx,None].expand((B,NS,len(lidx), 1)),dim=-1).squeeze(-1)
    lp_tok = x

    lp_next_token = (x.logsumexp(1) - math.log(NS))[:,lidx[-1]]
    lp_cond           = lp_tok.sum((1,2))


    lp_b = 0
    lp_b += lp_tok.sum((2,))  ## (B,NS) 
    lp_b += torch.einsum('bnh,bnh->bn', node_h_noise, (node_h_p+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.
    lp_b += torch.einsum('bnh,bnh->bn', 1-node_h_noise, (1-node_h_p+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.

    lp_b += math.log(0.5)*torch.ones_like(node_z_noise).sum(1)

    lp_b += -torch.einsum('bzn,bz->bn', node_z_noise, (node_z_p+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.
    lp_b += -torch.einsum('bzn,bz->bn', 1-node_z_noise, (1-node_z_p+EPS).log()) ### need to apply the entropy loss, because entropy is implicit on activation.

    lp_b = (lp_b - torch.log(torch.tensor(NS,device=device))).logsumexp(-1)
    if DEBUG:
        print(f'[lp_b]{lp_b.sum().item():.3f}')


        # lp_next_token = x = node_e[:,t:t+1].matmul(self.emb_vocab.T).log_softmax(-1)
    # lp_next_token = x = node_e_[:,t:t+1].matmul(self.emb_vocab.T).log_softmax(-1)

    # lp_cond           = torch.gather(lp_next_token,index=tok_external[:,t:t+1,None],dim=-1).squeeze(-1)
      ### (B, 1)
    xd_next.lp_ar = xbmd.lp_ar + lp_cond

    xd_next.lp_next_token = lp_next_token
    # print(xd_next.lp_next_token.shape)

    xd_next.lp_joint_bm = lp_b[:,None]


    ### update z 
    #node_ie_next   -> # (B,  M,  2L,  E)
    xd_next.node_h = node_h
    return xd_next

    breakpoint()
    

    ### need to sample on z to estimate lower bound for lp_internal
    def add_noise(lat,betasq):
      device = lat.device
      sigma = (1./betasq)**0.5
      noise = torch.normal(0., sigma, size=lat.shape, device=device)              
      return (lat+noise, noise)


    # z =

    
        
    inner_params = []
    def get_local_logp():
      #node_ie_exp        # (B,  K , H)
      node_h_noise, noise = add_noise(node_h, betasq_2)

      ### first layer to emit, last layer the final var
      ### (B,K-1,H)  (B,K-1,H,H)
      node_wke = torch.sigmoid( torch.einsum('bkh,bkhj->bkj', node_h_noise[:,1:,:], w_k[None,:,:,:].expand(B,K-1,H,H)) + b_k[None])
      node_wke = torch.cat([ 
          node_wke,
          torch.zeros((B,1,H),device=device) + 0.5,
          ],1)

      lp_internal = torch.sum(

                0.5* math.log(betasq/2/PI)
                - 0.5* betasq * torch.square( node_h_noise[:,:,:] - node_wke)
                
                - 0.5* math.log(betasq_2/2/PI)
                + 0.5*betasq_2 * torch.square( noise[:,:,:] ),
          axis=(1,2),
          )
      # (B, L, E)
    #   node_e = torch.tensordot( node_h_noise[:,0], (w_hle), 1) + b_hle[None]
      node_e = (torch.tensordot( node_h_noise[:,0], (w_hle), 1) + b_hle[None]).sigmoid()

      ### drop last node if is_mask_last==True
      x = torch.tensordot(node_e[:,0:t+1-is_mask_last ], emb_vocab_w.T,1).log_softmax(-1)
      x = torch.gather(x, index=tok_external[:, 0:t+1-is_mask_last,None],dim=-1).squeeze(-1)
      lp_external = x.sum(dim=1)



      #### estimating the importance of all possible branches
      opt_logp        = ( lp_internal + lp_external)
      return opt_logp,node_e    

    ### backprop on z=node_ie_exp
    # loss  = - opt_logp.sum()
    for i in range(inner_nstep):
      for x in inner_params:
        x.grad = None
      opt_logp = 0.
      for i in range(inner_nsample):
        opt_logp_diff, node_e = get_local_logp()
        opt_logp += opt_logp_diff
      opt_logp        = opt_logp / inner_nsample 

      loss = - opt_logp.sum()

      loss.backward(retain_graph=True)
      # loss.backward([node_ie_exp], retain_graph=True)
      for x in inner_params:
        x.data.sub_( inner_lr * x.grad.data )
        x.grad = None
      # print(f'loss={loss.detach().numpy()}')
    # [x.grad =N]

    for x in self.parameters():
      x.grad = None

    opt_logp_max       = opt_logp
    if is_ar_loss:
      ### score the options according to ability to predict the next
      ### score the external loss at the L=t
      ### (B,  M,  2L,  2K,  2L,  E)  
      ### (B,  M,   1,  2K,  2L,  V)  
      x = node_e[:,t:t+1].matmul(self.emb_vocab.T).log_softmax(-1)
      ## (B,1,V)
      xd_next.lp_next_token = x.squeeze(1)


      ### lp_per_batch
      ### (B, 1)
      lp_cond           = torch.gather(x,index=tok_external[:,t:t+1,None],dim=-1).squeeze(-1)

      ### (B, 1)
      xd_next.lp_ar = xbmd.lp_ar + lp_cond

    xd_next.lp_joint_bm = opt_logp
    ### update z 
    #node_ie_next   -> # (B,  M,  2L,  E)
    xd_next.node_h = node_h

    return xd_next          


  def get_init_dict(self, tok_external):      
    PI      = 3.1415926

    params = self.params
    device = self.device

    K = params.K
    E = params.E
    M = params.M
    T = params.T
    H = params.H
    Z = params.Z
    B,L = tok_external.shape
    EPS = params.EPS
    assert L<=T,(L,T)
    # tok_external     = tok_external                        ### (B, L)

    node_z    = torch.zeros((B,Z),device=device) +EPS                    ### (B, M)
    bm_dict_next = AttrDict(
      tok_external=tok_external,
      node_z = node_z,
      node_h = (torch.einsum('bz,hz->bh', node_z, self.whz) + self.bh[None]).sigmoid(),
      # node_h    = torch.zeros((B,H),device=device) + EPS,                     ### (B, M)
    #   lp_graph    = torch.zeros((B,M),device=device),                     ### (B, M)
    #   node_ie     = torch.zeros((B,M,2*L,E),device = device),             ### (B, M, 2*L, E)
    #   node_par    = torch.zeros((B,M,2*L),device = device).long() - 0,    ### (B, M, 2L)
    #   node_par_k  = torch.zeros((B,M,2*L),device = device).long() - 0,    ### (B, M, 2L)
      lp_joint_bm = torch.zeros((B,1),device=device)  , 
    #   node_h = torch.zeros((B,K,H),device=device,requires_grad=True),
      
      lp_ar            = torch.zeros((B,1),device=device),
      max_t = min(L,T),


    )
    return bm_dict_next

  def forward(self, tok_external, enc_inputs_len, return_attn=False,
    is_ar_loss=False,
  ):        
    bm_dict_next = self.get_init_dict(tok_external)
    max_t = bm_dict_next.max_t 
    
    if is_ar_loss:
        # if self.params.method=='elbo':
        #     min_t =  1
        #     max_t = max_t
        # else:

            min_t = max_t-1
            max_t = max_t
    else:
        min_t = max_t-1
        max_t = max_t

    for t in range(min_t,  max_t):
        #   print(f'[t={t}]')
        bm_dict = self._inner_loop( bm_dict_next, t, max_t, is_ar_loss=is_ar_loss)
        bm_dict_next = bm_dict
        

    if is_ar_loss:
      loss_bm = bm_dict_next.lp_ar
    else:
      loss_bm = bm_dict_next.lp_joint_bm
    return (loss_bm, bm_dict_next)


  def sample(self, 
    tok_external, t, 
    is_ar_loss=False,
  ):        
    '''
    ### tok_external: tensor of shape (B, L) 
    ### return_vals:
    ###   tok_external, bm_dict
    ### given the current tokens, sample the output

    '''
    tok_external = tok_external.clone()
    B, L = tok_external.shape
    V = self.params.V
    params = self.params

    tok_external = torch.cat([ 
      tok_external, 
      params.V-1 + torch.zeros((B,t),device=self.device).long()], dim=1)

    xd_next = self.get_init_dict(tok_external)
    max_t = tok_external.shape[1]

    xd_next.lp_next_token = None

    if self.params.method=='elbo':
        for t in range(0,  max_t):
            if t+1>=L+1:
                xd      = self._inner_loop( xd_next, t, max_t, is_ar_loss=True)
                ### (B,V)
                xp = xd.lp_next_token[:,None].exp().cumsum(dim=-1)
                idx = (xp > torch.rand_like(xp)).max(dim=-1)[1]
                tok_external[:,t:t+1] = idx
            else:
                xd      = self._inner_loop( xd_next, t, max_t, is_ar_loss=False)

    #     xd_next = xd
    return tok_external
    


  def loss_joint_lp(self, tok_external, enc_inputs_len=None, is_ar_loss=False):
    lp_joint_bm = self.forward(tok_external, enc_inputs_len, is_ar_loss=is_ar_loss)[0]
    loss = -( lp_joint_bm.logsumexp(-1).mean(0))
    # print(lp_joint_bm.shape)
    return loss




import time
from numpy import array


def main():  
  T = L = 20
  K = 10
  E = 11
  B = 12
  M = 13
  EPS = 1E-8



  ### test fitting with synthetic data
  params = AttrDict(dict(
    L = 15,
    K = 7,
    
    E = 11,
    H = 20,
    Z = 21,


    B = 3,
    M = 12,
    EPS = 1E-8,
    # is_ar_loss = True,
    # is_ar_loss = False,

    betasq           = 12. ,  ### less noise in posterior
    betasq_2         = 12.  ,

    inner_nsample = 3,  ### how many samples to be taken to estimate elbo
    inner_nstep   = 10,
    inner_lr      = 0.03,


  ))
  params.V = 30
  params.T = 60
  # device = None  

  if '--cpu' in sys.argv:
      device = torch.device("cpu") 

  else:
      device = torch.device("cuda:0") 
  model = SRBM(params,device)
  tok_ext = torch.randint(0, params.V,size=(B,params.L),device=device)
  # print()
  torch.manual_seed(0)
  tok_sampled = model.sample(tok_ext[:,:5], t = 5)

  print(tok_ext.shape)
  print(tok_sampled.shape)
  print(tok_ext[0])
  print(tok_sampled[0])

  loss  = model.loss_joint_lp(tok_ext)

  nstep = 100
  lr = 0.01
  optimizer = torch.optim.RMSprop(model.parameters(),lr=lr)
  # optimizer =
  data_iter = [tok_ext]
  def toc(t0=[time.time()]):
    _t0 = t0[0]
    t0[0] = t1 = time.time()
    return (t1 - _t0)
  toc()
  for stepp in range(nstep):
    for dataa in data_iter:
      optimizer.zero_grad()
      # loss  = model.loss_joint_lp(tok_ext,is_ar_loss=False)
      
      ### ar_loss = True
      ### comment: directly trainning auto-regression
      ### [step=99] loss=41.485 ar_loss=40.861 elapsed=2.78sec
  
      ### ar_loss = False 
      ### comment: superior and yields better ar_loss
      ### [step=99] loss=59.842 ar_loss=36.480 elapsed=2.67sec
      ### [step=99] loss=56.858 ar_loss=30.827 elapsed=2.61sec

      loss  = model.loss_joint_lp(tok_ext,is_ar_loss=False)
      # loss  = model.loss_joint_lp(tok_ext,is_ar_loss=True)      
      ar_loss = model.loss_joint_lp(tok_ext,is_ar_loss=True)
      
      loss.backward()
      optimizer.step()
      print(f"[step={stepp}] loss={loss.detach().cpu().numpy():.3f} ar_loss={ar_loss.detach().cpu().numpy():.3f} elapsed={toc():.2f}sec")
    
  print('[all_test_passed]')


  ### testing sampling from a fitted model
  
  breakpoint()
if __name__ == '__main__':

  main()
