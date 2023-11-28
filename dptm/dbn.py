import math
import torch
import torch.nn as nn

# from transformer.data import data_utils
# from transformer.data import data_utils


# def get_attn_pad_mask(seq_q, seq_k):
#     assert seq_q.dim() == 2 and seq_k.dim() == 2
#     b_size, len_q = seq_q.size()
#     b_size, len_k = seq_k.size()
#     pad_attn_mask = seq_k.data.eq(data_utils.PAD).unsqueeze(1)  # b_size x 1 x len_k
#     return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k

def expand_graph_proposals(params, node_par, node_par_k, t):
  '''
  expand graph to cover proposals

  # node_par_next  = None     ### (B, M, 2L)
  # node_par_k_next= None     ### (B, M, 2L)
  
  #node_par_prop        (B,  M,  2L,  2K,  2L)
  #node_par_k_prop      (B,  M,  2L,  2K,  2L)
  #opt_prior_cond       (B,  M,  2K,  2L)

  (node_par_prop,  
    node_par_k_prop, 
    opt_prior_cond) = expand_graph_proposals(
      self.params, node_par, node_par_k, t)  
  '''
  # L      = params.L
  K      = params.K
  E      = params.E
  EPS = params.EPS
  device = node_par.device
  L =  node_par.shape[2]//2
  B = len(node_par)
  M = params.M
  # B = params.B

  ### npar points to current parent of this node
  ### store the parent information about each node
  ### update the topology
  node_par           # (B,  M,  2L,)  ## the index of parent
  # node_par_exp     # (B,  M,  2L,  O=2L)  ## the index of parent
  node_par_exp       = node_par[:,:,:,None,]
  node_par_exp       = node_par_exp.expand(*(B,M,2*L,2*L)).clone()
  node_sib           = torch.arange(2*L, device=device)[None,None,None,:]
  
  sib_idx_1 = torch.arange(2*L, device=device)[:,None]
  sib_idx_2 = sib_idx_1
  # sib_idx_1 = torch.arange(2*L, device=device)[None,:].repeat((2*L,1))
  # torch.arange(2*L, device=device)[:,None].repeat((1,  1))
  # sib_idx_2 = torch.arange(2*L, device=device)[:,None].repeat((1,  1))

  node_par_exp_c     = node_par_exp.clone()
  node_par_exp_c[:,:,sib_idx_1,sib_idx_2] = (L+t)  ### connect the sibling to the new internal node
  node_par_exp_c[:,:,t,:]                 = (L+t)  ### connect the new external to the new internal node.
  if t>0:
    ### connect the new internal node to the 
    node_par_exp_c[:,:,L+t,:]   = node_par_exp[:,:, sib_idx_1,sib_idx_2].squeeze(-1)


  node_par_k         # (B,  M,  2L,  1,  1, )  ## the type of connection to parent
  node_par_k         # (B,  M,  2L,  2K,  2L, )  ## the type of connection to parent
  node_par_k_exp     = node_par_k[:,:,:,None,None].expand(*(B,M,2*L,2*K,2*L))

  k_new = torch.arange(2*K,device=device)[:,None].expand(*(2*K,2*L))
  k_sib = (k_new + K) % (2*K) 


  ## (K,L) matrix
  idx_sib_x  = torch.arange(2*L, device=device)[None].expand(*(2*K,2*L))
  idx_sib_k  = torch.arange(2*K, device=device)[:,None].expand(*(2*K,2*L))
  idx_t  = (idx_sib_x * 0) + t 

  node_par_k_exp_c = node_par_k_exp.clone()
  ### sibling node
  node_par_k_exp_c[:,:, idx_sib_x, idx_sib_k, idx_sib_x] = k_sib   
  ### new external node
  node_par_k_exp_c[:,:, idx_t,     idx_sib_k, idx_sib_x] = k_new
  ### new internal node <- inherit the parent_k of sibling
  node_par_k_exp_c[:,:, idx_t+L,   idx_sib_k, idx_sib_x] = node_par_k_exp[:,:,idx_sib_x, idx_sib_k, idx_sib_x]
  
  ###
  node_par_exp_c     = node_par_exp_c[:,:,:,None,:].expand(*(B,M,2*L,2*K,2*L))

  ### only allow joining to the active nodes
  opt_prior_cond          = torch.cat([ 
      torch.arange(L, device=device)<= max(0, t-1), 
      (torch.arange(L, device=device)<= t-1) & (torch.arange(L, device=device)>0),   
      ### first hidden node is root and irreplacible
      ], 
        dim=0).double()[None].expand(*(2*K,2*L))            
  ## (2K,2L)
  shape = opt_prior_cond.shape
  # opt_prior_cond          = (opt_prior_cond+ EPS).reshape((-1)).log_softmax(0).reshape(opt_prior_cond.shape)
  opt_prior_cond          = (opt_prior_cond+ EPS)
  opt_prior_cond          = (opt_prior_cond / opt_prior_cond.sum()).log()
  #  (opt_prior_cond+ EPS).reshape((-1)).log_softmax(0).reshape(opt_prior_cond.shape)
  ## (B, M, 2K, 2L)
  opt_prior_cond          = opt_prior_cond[None,None,:,:,]

  return (node_par_exp_c, node_par_k_exp_c, opt_prior_cond)

def select_bmkl(array, idx_array):
  '''
  idx_array reshaped to (B,M)
  shape of (B, M, 2K, 2L, ...) -> (B,M,...)


  node_ie_next    = select_bm(node_ie_exp, max_m_idx)
  ### update parent and k
  node_par_next   = select_bm(node_par_exp, max_m_idx)
  node_par_k_next = select_bm(node_par_k_exp, max_m_idx)
  
  '''
  _,_,K2,L2 = array.shape[:4]
  B,M = idx_array.shape
  device = array.device
  max_m      = idx_array

  factor     = max_m // (K2*L2)
  max_m      = max_m  % (K2*L2)
  idx_m      = factor

  factor     = max_m // (L2)
  max_m      = max_m  % (L2)
  idx_k      = factor

  idx_l      = max_m
  idx_b      = torch.arange(B, device=device)[:,None].repeat((1,M))
  sel = array[idx_b, idx_m, idx_k, idx_l]
  return sel


from attrdict import AttrDict
# class Decoder(nn.Module):
PI = 3.1415926
class DBN(nn.Module):
  '''
  Dynamic probabilistic tree models
  '''
  def __init__(self, 
      params,
      device=None,
      # n_layers, 
      # d_k, d_v, d_model, d_ff, n_heads,
      # max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False,
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
      # self.params = AttrDict()
      if device is None:
        device = torch.device('cpu')
      self.device = device

      L = params.L

      # self.emb_vocab  = nn.Embedding(V, E, padding_idx=data_utils.PAD, device=device)
      self.emb_vocab  = nn.Parameter(nn.Linear(E, V,).weight.to(device))
      #  (V, E)
      self.w_k        = nn.Parameter(nn.Linear(1,K*H*H).weight.reshape((K,H,H))[1:].to(device))
      self.b_k        = nn.Parameter(nn.Linear(1,(K-1)*H).weight.reshape((K-1,H)).to(device))
      self.w_hle      = nn.Parameter(nn.Linear(1,H*L*E).weight.reshape((H,L,E)).to(device))
      self.b_hle      = nn.Parameter(nn.Linear(1,L*E).weight.reshape((L,E)).to(device))
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

    node_h      = xbmd.node_h

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

    inner_params = [node_h]

    ### need to sample on z to estimate lower bound for lp_internal
    def add_noise(lat,betasq):
      device = lat.device
      sigma = (1./betasq)**0.5
      noise = torch.normal(0., sigma, size=lat.shape, device=device)              
      return (lat+noise, noise)

    def get_local_logp():
      #node_ie_exp        # (B,  K , H)
      node_h_noise, noise = add_noise(node_h, betasq_2)

      ### first layer to emit, last layer the final var
      ### (B,K-1,H)  (B,K-1,H,H)
    #   node_wke = torch.einsum('bkh,bkhj->bkj', torch.tanh(node_h_noise[:,1:,:]), w_k[None,:,:,:].expand(B,K-1,H,H)) + b_k[None]
    #   node_wke = torch.einsum('bkh,bkhj->bkj', torch.sigmoid(node_h_noise[:,1:,:]), w_k[None,:,:,:].expand(B,K-1,H,H)) + b_k[None]
      node_wke = torch.sigmoid( torch.einsum('bkh,bkhj->bkj', node_h_noise[:,1:,:], w_k[None,:,:,:].expand(B,K-1,H,H)) + b_k[None])
      node_wke = torch.cat([ 
          node_wke,
          torch.zeros((B,1,H),device=device),
          ],1)
      lp_internal = torch.sum(

                0.5* math.log(betasq/2/PI)
                - 0.5* betasq * torch.square( node_h_noise[:,:,:] - node_wke)
                
                - 0.5* math.log(betasq_2/2/PI)
                + 0.5*betasq_2 * torch.square( noise[:,:,:] ),
          axis=(1,2),
          )
      # (B, L, E)
      node_e = torch.tensordot( node_h_noise[:,0], (w_hle), 1) + b_hle[None]

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
    B,L = tok_external.shape
    assert L<=T,(L,T)
    # tok_external     = tok_external                        ### (B, L)

    bm_dict_next = AttrDict(
      tok_external=tok_external,
    #   lp_graph    = torch.zeros((B,M),device=device),                     ### (B, M)
    #   node_ie     = torch.zeros((B,M,2*L,E),device = device),             ### (B, M, 2*L, E)
    #   node_par    = torch.zeros((B,M,2*L),device = device).long() - 0,    ### (B, M, 2L)
    #   node_par_k  = torch.zeros((B,M,2*L),device = device).long() - 0,    ### (B, M, 2L)
      lp_joint_bm = torch.zeros((B,1),device=device)  , 
      node_h = torch.zeros((B,K,H),device=device,requires_grad=True),
      
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
        min_t =  0
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
    
    for t in range(0,  max_t):
        if t+1>=L+1:
          xd      = self._inner_loop( xd_next, t, max_t, is_ar_loss=True)
          ### (B,V)
          xp = xd.lp_next_token[:,None].exp().cumsum(dim=-1)
          idx = (xp > torch.rand_like(xp)).max(dim=-1)[1]
          tok_external[:,t:t+1] = idx
        else:
          xd      = self._inner_loop( xd_next, t, max_t, is_ar_loss=False)

        xd_next = xd
    return tok_external
    


  def loss_joint_lp(self, tok_external, enc_inputs_len=None, is_ar_loss=False):
    lp_joint_bm = self.forward(tok_external, enc_inputs_len, is_ar_loss=is_ar_loss)[0]
    loss = -( lp_joint_bm.logsumexp(-1).sum(0))
    return loss




import time
from numpy import array


if __name__ == '__main__':

  T = L = 20
  K = 10
  E = 11
  B = 12
  M = 13
  EPS = 1E-8

#   x = torch.rand((B,M,2*K,2*L,2*L,E))
#   _, y = torch.rand((B,M,2*K,2*L)).reshape((B,-1)).topk(k=10,dim=-1)
#   xx = select_bmkl(x,y)
#   print(xx.shape)

#   x = torch.rand((B,M,2*K,2*L,2*L,E,3))
#   xx = select_bmkl(x,y)
#   print(xx.shape)


#   params = AttrDict(dict(
#     L = 5,
#     K = 7,
#     E = 11,
#     B = 3,
#     M = 13,
#     EPS = 1E-8,

#   ))

#   ### npar (B,M,2L)
#   t = 3
#   B = params.B
#   ## 0, 1, 2 must has parents
#   ## 0, 1 mst 
#   node_par = torch.tensor([
#     [ 5+1, 5+1,  5+2, 0, 0,
#       0,   5+2,  5+0, 0, 0,
#     ]
#   ])[None].repeat((B,1,1))

#   node_par_k = torch.tensor([
#     [ 1,   6,   2,  -1, -1,
#       -1,   7,   3,  -1, -1,
#     ]
#   ])[None].repeat((B,1,1))
#   print(node_par.shape)
#   print(node_par_k.shape)
#   (npar, npark, opt_prior) = expand_graph_proposals(params, node_par, node_par_k, t=3, )


#   idx_l = list(range(0,4)) + list(range(5,9))

#   val = torch.cat([    node_par[0,0][None], node_par[0,0][None]*0+1, npar[0,0,:,0,idx_l].T],dim=0).numpy()
#   print(val)

#   '''
#   array([[6, 6, 7, 0, 0, 0, 7, 5, 0, 0],
#          [8, 6, 7, 8, 0, 0, 7, 5, 6, 0],
#          [6, 8, 7, 8, 0, 0, 7, 5, 6, 0],
#          [6, 6, 8, 8, 0, 0, 7, 5, 7, 0],
#          [6, 6, 7, 8, 0, 0, 7, 5, 0, 0]])
  
#   '''  
#   expected  = array([
#        [6, 6, 7, 0, 0, 0, 7, 5, 0, 0],
#        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#        [8, 6, 7, 8, 0, 0, 7, 5, 6, 0],
#        [6, 8, 7, 8, 0, 0, 7, 5, 6, 0],
#        [6, 6, 8, 8, 0, 0, 7, 5, 7, 0],
#        [6, 6, 7, 8, 0, 0, 7, 5, 0, 0],
#        [6, 6, 7, 8, 0, 8, 7, 5, 0, 0],
#        [6, 6, 7, 8, 0, 0, 8, 5, 7, 0],
#        [6, 6, 7, 8, 0, 0, 7, 8, 5, 0],
#        [6, 6, 7, 8, 0, 0, 7, 5, 0, 0]])
#   assert (val == expected).all()


#   val = torch.cat([    node_par_k[0,0][None], node_par_k[0,0][None]*0+1, npark[0,0,:,0,idx_l].T],dim=0).numpy()
#   expected = array([
#        [ 1,  6,  2, -1, -1, -1,  7,  3, -1, -1],
#        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
#        [ 7,  6,  2,  0, -1, -1,  7,  3,  1, -1],
#        [ 1,  7,  2,  0, -1, -1,  7,  3,  6, -1],
#        [ 1,  6,  7,  0, -1, -1,  7,  3,  2, -1],
#        [ 1,  6,  2,  0, -1, -1,  7,  3, -1, -1],
#        [ 1,  6,  2,  0, -1,  7,  7,  3, -1, -1],
#        [ 1,  6,  2,  0, -1, -1,  7,  3,  7, -1],
#        [ 1,  6,  2,  0, -1, -1,  7,  7,  3, -1],
#        [ 1,  6,  2,  0, -1, -1,  7,  3, -1, -1]])
#   print(val)
#   assert (val == expected).all()


#   print(
#     torch.cat([    node_par_k[0,0][None], node_par_k[0,0][None]*0+1, npark[0,0,:,1,idx_l].T],dim=0).numpy()
#   )

#   val = (opt_prior.exp()[0,0,:,idx_l] > 0.01).long().numpy()
#   expected = array([
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0],
#        [1, 1, 1, 0, 0, 1, 1, 0]])

#   print(val)
#   assert (val == expected).all()


  ### test fitting with synthetic data
  params = AttrDict(dict(
    L = 15,
    K = 7,
    
    E = 11,
    H = 20,


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
  device = torch.device("cuda:0") 
  model = DBN(params,device)
  tok_ext = torch.randint(0, params.V,size=(B,params.L),device=device)
  # print()
  torch.manual_seed(0)
  tok_sampled, xdict = model.sample(tok_ext, t = 5)

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
