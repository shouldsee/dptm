
import torch.nn as nn
import torch

import math
from attrdict import AttrDict
class ZBML3(nn.Module):
  def __init__(self, 
      params,
      device=None,
      ):
      super().__init__()

      self.params = params

      V = params.V
      if not isinstance(V,int):
        V = math.prod(V)
      FV = V  ### flat V



      if device is None:
        device = torch.device('cpu')
      self.device = device

      params.Z = params.E + 1

      # params.T = 3
      params.K = params.Z - 1
    #   params.M = 30
      params.M

      T = params.T
      E = params.E
      Z = params.Z

      
    #   self.w_enc = nn.Linear(FV,E).to(device)
      self.w_dec         = nn.Linear(E,FV).to(device)
      self.b_scale_prior = nn.Parameter(torch.tensor(0.,device=device))
      self.b_scale_post  = nn.Parameter(torch.tensor(0.,device=device))

      self.b_scale_out   = nn.Parameter(torch.tensor(0.,device=device))
      self.b_loc_prior   = nn.Parameter( nn.Linear(Z,1).weight.reshape((Z,)).to(device))


      ## (T,Z,L)
      self.whz           = nn.Parameter(nn.Linear(Z,FV*T).weight.reshape(T,Z,FV).to(device))
      self.bias          = nn.Parameter(nn.Linear(Z,1).bias.to(device))
      self.prior_tzz     = nn.Parameter(nn.Linear(Z*Z*T,1).weight.reshape(T,Z,Z).to(device))
      


  def sample(self, 
    shape, tok):
    shape = tuple(shape)


    # w_enc = self.w_enc
    w_dec = self.w_dec
    b_scale_out = self.b_scale_out
    b_scale_post = self.b_scale_post
    b_loc_prior = self.b_loc_prior
    b_scale_prior = self.b_scale_prior    
    # self.b_loc_prior



    # B = 
    # ### (T,Z1,Z2)
    prior_tzz_log = self.prior_tzz.log_softmax(-1)

    # ### (B,M,K,T,Z2)
    # lp_prior = lp_prior[idx_t, xkz_parent, :]
    
    # prior = torch.distributions.Categorical(logits=lp_prior)
    shape_orig = shape
    shape = (math.prod(shape),)
    B = shape[0]

    T = self.params.T
    device =self.device
    xbz = torch.zeros(shape+(T,),device=device).long()
    xbzt = torch.zeros(shape+(1,),device=device).long()

    for t in range(T):
        ## (B,1)
        lprior = prior_tzz_log[t][xbzt,:]
        prior = torch.distributions.Categorical(logits=lprior)
        xbzt = prior.sample()
        xbz[:,t:t+1]= xbzt


    # whz  = self.whz ## (T,Z,L)

    ### (B,T,L)
    idx_t = torch.arange(T)[None,:].expand(B,T)        


    nl = self.whz[idx_t, xbz,:]
    # L = self.params.L
    L = 28**2
    sep = (L //T)+1

    mask = torch.arange(L,device=device)[None,:]
    it   = torch.arange(T,device=device)[:,None]
    mask = (mask >= it * sep) & (mask < (it+1)*sep)
    # mask = 
    nl = nl * mask[None,]


    ### (B,L)
    # loc_pred = (nl.sum(-2)+self.bias[None,:,])
    loc = (nl.sum(-2)+self.bias[None,:,])
    loc = loc.reshape(shape_orig+tuple(self.params.V))
    return loc


    # loc = w_dec(z_noise)
    loc = torch.rand(shape+tuple(self.params.V))
    # loc = loc.reshape(shape+tuple(self.params.V))
    return loc

  def generate(self, 
    tok_external, t, 
    is_ar_loss=False,
  ):        
    pass

  def forward(self, tok_emb, h0=None):
    pass   

  def loss_joint_lp(self, dat):
    img,lab = dat

    whz  = self.whz ## (T,Z,L)
    bias = self.bias

    # w_enc = self.w_enc
    w_dec = self.w_dec
    b_scale_out   = self.b_scale_out
    b_scale_post  = self.b_scale_post
    b_loc_prior   = self.b_loc_prior
    b_scale_prior = self.b_scale_prior

    params        = self.params

    img_flat = img.reshape((len(img),-1))
    node_bx  = img_flat
    # B, L = img.shape

    B = img.size()[0]
    E = self.params.E
    T = params.T
    M = params.M


    Z = params.Z
    K = params.K
    device = self.device
    # xmz = None ### (B,M,T)
    # wzl = None  ## (T,Z,L)

    xmz = torch.zeros((B,M,T),device=device).long()
    # wzl = self.wzl
    # nstep = 1
    idx_bm_b = torch.arange(B)[:,None].expand(B,M)
    idx_bm_m = torch.arange(M)[None,:].expand(B,M)
    idx_bmkt = torch.arange(T,device=device)[None,None,None].expand(B,M,K,T)

    lp_prior_0 = self.prior_tzz.log_softmax(-1)
    nstep = params.inner_nstep
    L = 28**2
    sep = (L //T)+1
    mask = torch.arange(L,device=device)[None,:]
    it = torch.arange(T,device=device)[:,None]
    mask_tl = (mask >= it * sep) & (mask < (it+1)*sep)

    # with torch.no_grad():
    if 1:
      for nstepp in range(nstep):
          for t in range(T):
              ### (B, M, K, T)
              # xkz = xmz.unsqueeze(2).expand(B,M,K,T)
              xkz = xmz.unsqueeze(2).expand(B,M,K,T)
              # .detach()

              ### ()
              idx_k = torch.arange(K,device=device)
              # xkz[idx_b,idx_m,idx_k,idx_t]
              xkz[:,:,:,t][:,:,idx_k]=idx_k


              ### (B,M,K,T,L)        
              nl = whz[idx_bmkt,xkz,:]

              nl = nl * mask_tl[None,None,None].detach()

              ### (B,M,K,L)
              ### (K,M,B,L)
              loc_pred = (nl.sum(-2)+bias[None,None,None, :,]).transpose(0,2)
              mod_pred = torch.distributions.normal.Normal(loc=loc_pred,scale=b_scale_out.exp())
              lp_external = mod_pred.log_prob(node_bx.detach()).sum(-1)
              lp_external = lp_external.transpose(0,-1)

              # loc_pred = (nl.sum(-2)+bias[None,None,None, :,])            
              # ### (B,M,K,L)
              # mod_pred = torch.distributions.normal.Normal(loc=loc_pred,scale=b_scale_out.exp())
              # ## node_bx
              # lp_external = mod_pred.log_prob(node_bx[:,None,None,:].detach()).sum(-1)
              ## (B,M,K)
              # lp_external = lp_external.transpose(0,-1)

              ### (B,M,K,T)
              xkz_parent = torch.cat([torch.zeros((B,M,K,1),device=device).long(),
                  xkz[:,:,:,:-1]
              ],dim=-1)
              
              ### (T,Z1,Z2)
              ### (B,M,K,T,Z2)
              lp_prior = lp_prior_0[idx_bmkt, xkz_parent, :]
              prior = torch.distributions.Categorical(logits=lp_prior)

              ### (B,M,K,T)
              ### (B,M,K)
              lp_internal = prior.log_prob(xkz).sum(-1)
              lp_total = lp_external + lp_internal



              ### (B,M)
              # idx_b = torch.arange(B)[:,None,].expand(B,M)
              # idx_t = torch.arange(T)[None,None,:].expand(B,M*K,T)

              ### for each of the M model, sample from K distrib
              ### sample (B,M)
              # p_total = lp_total.softmax(-1)
              # idx_k = (torch.rand_like(p_total[:,:,-1:]) < p_total.cumsum(-1)).max(-1)[1]

              gibbs = torch.distributions.Categorical(logits=lp_total)
              # (B,M)
              idx_k = gibbs.sample()

              lp_m = torch.gather(lp_total,index=idx_k.unsqueeze(-1),dim=-1).squeeze(-1)
              ### (B,M,K,T)
              xmz = xkz[idx_bm_b,idx_bm_m,idx_k,:]

              # idx_b = torch.arange(B)[:,None,].expand(B,M)
              # # idx_t = torch.arange(T)[None,None,:].expand(B,M*K,T)
              # ## select top M from M,K
              # xkz_bmk = xkz.reshape((B,-1,T))
              # lp_m, idx = lp_total.reshape((B,-1)).topk(M,dim=-1)
              # ## (B,M,T)
              # xmz = xkz_bmk[idx_b, idx,:]

              ### (B,M)
              lp_m


              if nstepp+1 == nstep and t+1==T:
                pass
              else:
                lp_total = lp_total.detach()
                xmz = xmz.detach()

    EPS = params.EPS
    ### modify posterior to estimate elbo
    # lp_post = lp_m.log_softmax(-1)
    lse = (lp_m).logsumexp(-1)
    # print(lse.shape)
    # xd = self.get_init_dict(dat)    
    loss = - lse.mean(0)


    # loss = -( lp_joint_bm.logsumexp(-1).sum(0))
    return loss

    '''
    [training_epoch=0]:   1%|         | 99/10000 [00:13<21:36,  7.64it/s]validation loss: [7721.849, 7724.963]
TRAINING loss: [6193.791, 6189.467]
[training_epoch=0]:   2%|▏       | 195/10000 [00:26<21:22,  7.65it/s]validation loss: [6043.585, 6060.63]
TRAINING loss: [5307.924, 5312.458]
[training_epoch=0]:   3%|▏       | 299/10000 [00:40<21:10,  7.64it/s]validation loss: [5531.236, 5525.326]
TRAINING loss: [4687.746, 4687.602]
[training_epoch=0]:   4%|▎       | 395/10000 [00:52<20:58,  7.63it/s]validation loss: [4635.46, 4641.023]
TRAINING loss: [4422.282, 4424.258]
[training_epoch=0]:   5%|▍       | 499/10000 [01:06<20:46,  7.62it/s]validation loss: [4010.281, 4023.137]
TRAINING loss: [4487.747, 4480.412]

    '''

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