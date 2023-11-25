import math
import torch
import torch.nn as nn

from transformer.data import data_utils


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(data_utils.PAD).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k

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
  L      = params.L
  K      = params.K
  E      = params.E
  EPS = params.EPS
  device = node_par.device

  ### npar points to current parent of this node
  ### store the parent information about each node
  ### update the topology
  node_par           # (B,  M,  2L,)  ## the index of parent
  # node_par_exp     # (B,  M,  2L,  O=2L)  ## the index of parent
  node_par_exp       = node_par[:,:,:,None,]
  node_par_exp       = node_par_exp.repeat((1,1,1,2*L)).clone()
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
  node_par_k_exp     = node_par_k[:,:,:,None,None].repeat((1,1,1,2*K,2*L))

  k_new = torch.arange(2*K,device=device)[:,None].repeat((1,2*L))
  k_sib = (k_new + K) % (2*K) 


  ## (K,L) matrix
  idx_sib_x  = torch.arange(2*L, device=device)[None].repeat((2*K,1))
  idx_sib_k  = torch.arange(2*K, device=device)[:,None].repeat((1,2*L))
  idx_t  = (idx_sib_x * 0) + t 

  node_par_k_exp_c = node_par_k_exp.clone()
  ### sibling node
  node_par_k_exp_c[:,:, idx_sib_x, idx_sib_k, idx_sib_x] = k_sib   
  ### new external node
  node_par_k_exp_c[:,:, idx_t,     idx_sib_k, idx_sib_x] = k_new
  ### new internal node <- inherit the parent_k of sibling
  node_par_k_exp_c[:,:, idx_t+L,   idx_sib_k, idx_sib_x] = node_par_k_exp[:,:,idx_sib_x, idx_sib_k, idx_sib_x]
  
  ###
  node_par_exp_c     = node_par_exp_c[:,:,:,None,:].repeat((1,1,1,2*K,1))

  ### only allow joining to the active nodes
  opt_prior_cond          = torch.cat([ 
      torch.arange(L, device=device)<= max(0, t-1), 
      (torch.arange(L, device=device)<= t-1) & (torch.arange(L, device=device)>0),   
      ### first hidden node is root and irreplacible
      ], 
        dim=0).double()[None].repeat((2*K,1))            
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
class DPT(nn.Module):
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
        V = params.V
        K = params.K
        E = params.E
        M = params.M
        T = params.T
        # self.params = AttrDict()
        if device is None:
          device = torch.device('cpu')
        self.device = device

        # self.emb_vocab  = nn.Embedding(V, E, padding_idx=data_utils.PAD, device=device)
        self.emb_vocab  = nn.Linear(E, V,).weight.to(device)
        #  (V, E)
        self.w_k        = nn.Linear(2*K, E*E).weight.reshape((2*K,E,E)).to(device)
        #  (2K, E,  E)

        ### precicision squared of gaussian
        # self.betasq     = torch.tensor(1.0).to(device)
        self.betasq     = (1.0)


    def loss_joint_lp(self, tok_external, enc_inputs_len=None):
      lp_joint_bm = self.forward(tok_external, enc_inputs_len)[0]
      loss = -( lp_joint_bm.logsumexp(-1).sum(0))
      return loss
          

    def forward(self, tok_external, enc_inputs_len, return_attn=False):
        '''
        need to find the top m expansion

        enc_inputs      (B, L)
        enc_outputs     (B, L, E)

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
        
        PI      = 3.1415926

        params = self.params
        device = self.device

        K = params.K
        E = params.E
        M = params.M
        T = params.T


        B,L = tok_external.shape
        assert L<=T,(L,T)

        nsample     = 3  ### how many samples to be taken to estimate elbo
        inner_nstep = 10
        inner_lr    = 0.03


        tok_external     = tok_external                        ### (B, L)

        w_k_comb         = self.w_k                            #  (2K, E,  E)
        emb_vocab_w      = self.emb_vocab                      ### (V, E)
        betasq           = self.betasq        ### invert of sigma^2  
        betasq           = 1.  ### less noise in posterior
        betasq_2         = 10. 
        

        lp_graph_next    = torch.zeros((B,M),device=device)                     ### (B, M)
        node_ie_next     = torch.zeros((B,M,2*L,E),device = device)             ### (B, M, 2*L, E)
        node_par_next    = torch.zeros((B,M,2*L),device = device).long() - 0    ### (B, M, 2L)
        node_par_k_next  = torch.zeros((B,M,2*L),device = device).long() - 0    ### (B, M, 2L)
        lp_joint_bm_next = torch.zeros((B,M),device=device)   


        for t in range(0, min(L,T)):
            
            ### initialise from last state
            lp_graph    = lp_graph_next
            node_ie     = node_ie_next
            node_par    = node_par_next
            node_par_k  = node_par_k_next
            lp_joint_bm = lp_joint_bm_next

            ### t = 0 -> infer the first node, 1 obs + 0 hidden 1 choice
            ### t = 1 -> infer the 2nd node,   1 obs + 0 hidden        1 choice
            ### t = 2 -> infer the 3nd node.   2 obs + 1 hidden node   3 choice 2T-1

            ### beam search to update the node representation

            ### expand (2K*2L) options
            #node_par_prop        (B,  M,  2L,  2K,  2L)
            #node_par_k_prop      (B,  M,  2L,  2K,  2L)
            #opt_prior_cond       (B,  M,  2K,  2L)
            (node_par_prop,  
              node_par_k_prop, 
              opt_prior_cond) = expand_graph_proposals(
                self.params, node_par, node_par_k, t)

            #node_ie            # (B,  M,  2L,  E)            
            #node_ie_exp        # (B,  M,  2L,  2K,  2L,  E)
            node_ie_exp     = node_ie[:,:,:,None,None].repeat((1,1,1, 2*K, 2*L,1)).clone()
            node_ie_exp     = torch.tensor(node_ie_exp,device=device,requires_grad=True)
            
            #### getting the 
            #node_ie_par_prop     # (B,  M,  2L,  2K,  2L, E)  
            node_ie_exp

            ### need to sample on z to estimate lower bound for lp_internal
            def add_noise(lat,betasq):
              sigma = (1./betasq)**0.5
              noise = torch.normal(0., sigma, size=lat.shape, device=device)              
              return lat+noise,noise

            opt_logp  = 0.

            lp_graph_prop   = opt_prior_cond + lp_graph[:,:,None,None,]
            for i in range(nsample):

              #node_ie_exp        # (B,  M,  2L,  2K,  2L,  E)
              node_ie_exp_noise, noise = add_noise(node_ie_exp, betasq_2)
              node_ie_par_prop   = torch.gather(
                node_ie_exp, 
                index=node_par_prop[:,:,:,:,:,None].repeat((1,1,1,1,1,E)), dim=2)
              #node_ie_w_k_prop     # (B,  M,  2L,  2K,  2L, E)  
              node_ie_w_k_prop   = w_k_comb[node_par_k_prop,:,:]

              ###                # (B,  M,  2L,  2K,  2L,  E)  
              node_wke_prop    = torch.einsum("bmikje,bmikjef->bmikjf",
                  node_ie_par_prop,
                  node_ie_w_k_prop,
                  )

              
              ### set root node to center at zero 
              node_wke_prop[:,:,T:T+1] = 0.

              lp_internal  = torch.sum( 
                          0.5* math.log(betasq/2/PI)
                          - 0.5* betasq * torch.square( 
                          node_ie_exp_noise[:,:, T:T+t+1] 
                          - node_wke_prop[:,:,T:T+t+1]
                          )

                          - 0.5* math.log(betasq_2/2/PI)
                          + 0.5*betasq_2 * torch.square(noise[:,:,T:T+t+1])
                          ,axis=(2,5)
              )           
              # lp_internal = node_ie_exp.square().sum(axis=(2,5))
              # (B,  M,  2K,  2L)  


              ### scoring external nodes
              # (B,  M,  t+1,  2K,  2L,  V)  
              is_mask_last = False
              is_mask_last = int(is_mask_last)
              ### drop last node if is_mask_last==True
              x = torch.tensordot(node_wke_prop[:,:, 0:t+1-is_mask_last ], emb_vocab_w.T,1).log_softmax(-1)
              tok_external_exp = tok_external[:,None, 0:t+1-is_mask_last,None,None,None].repeat((1,M,1,2*K,2*L,1))
              x = torch.gather(x, index=tok_external_exp,dim=-1).squeeze(-1)
              lp_external = x.sum(dim=2)
              # (B,  M,  2K,  2L)  

              ### if last is masked, we can use ar loss to fit the model


              ### [TBC] adding the emission proba
              ### lp_graph            # (B,  M)
              ### lp_graph_prop       # (B,  M,  2K,  2L,)
              ### opt_prior_cond      log P( g2|g1 )
              ### lp_external         log P( {O}  | g2)
              ### lp_internal         log P( {Iz} | g2)

                          
              #### estimating the importance of all possible branches
              opt_logp        += ( lp_internal + lp_external + lp_graph_prop)

            opt_logp        = opt_logp / nsample 
            

            ### backprop on z=node_ie_exp
            loss  = - opt_logp.sum()
            for i in range(inner_nstep):
              loss.backward(retain_graph=True)
              # loss.backward([node_ie_exp], retain_graph=True)
              for x in [node_ie_exp]:
                x.data.sub_( inner_lr * x.grad.data )
                x.grad = None
              # print(f'loss={loss.detach().numpy()}')

            opt_logp_max       = opt_logp

            ### (B, M)
            lp , max_m_idx  = opt_logp_max.reshape((B, M*(2*K)*(2*L) )).topk( k=M, axis=1)
            lp_joint_bm_next = lp

            ### update z 
            #node_ie_exp       # (B,  M,  2L,  2K,  2L,  E)
            #node_ie_next   -> # (B,  M,  2L,  E)
            lp_graph_next   = select_bmkl(lp_graph_prop,   max_m_idx)
            node_ie_next    = select_bmkl(node_ie_exp.transpose(2,3).transpose(3,4), max_m_idx)

            ### update parent and k
            node_par_next   = select_bmkl(node_par_prop.transpose(2,3).transpose(3,4), max_m_idx)
            node_par_k_next = select_bmkl(node_par_k_prop.transpose(2,3).transpose(3,4), max_m_idx)

            # print(lp_graph_next.shape)
            # print(node_ie_next.shape)
            # print(node_par_next.shape)
            # print(node_par_k_next.shape)


        return (lp_joint_bm_next,( lp_graph_next, node_ie_next, node_par_next, node_par_k_next))


from numpy import array


if __name__ == '__main__':

  T = L = 20
  K = 10
  E = 11
  B = 12
  M = 13
  EPS = 1E-8

  x = torch.rand((B,M,2*K,2*L,2*L,E))
  _, y = torch.rand((B,M,2*K,2*L)).reshape((B,-1)).topk(k=10,dim=-1)
  xx = select_bmkl(x,y)
  print(xx.shape)

  x = torch.rand((B,M,2*K,2*L,2*L,E,3))
  xx = select_bmkl(x,y)
  print(xx.shape)


  params = AttrDict(dict(
    L = 5,
    K = 7,
    E = 11,
    B = 3,
    M = 13,
    EPS = 1E-8,

  ))

  ### npar (B,M,2L)
  t = 3
  B = params.B
  ## 0, 1, 2 must has parents
  ## 0, 1 mst 
  node_par = torch.tensor([
    [ 5+1, 5+1,  5+2, 0, 0,
      0,   5+2,  5+0, 0, 0,
    ]
  ])[None].repeat((B,1,1))

  node_par_k = torch.tensor([
    [ 1,   6,   2,  -1, -1,
      -1,   7,   3,  -1, -1,
    ]
  ])[None].repeat((B,1,1))
  print(node_par.shape)
  print(node_par_k.shape)
  (npar, npark, opt_prior) = expand_graph_proposals(params, node_par, node_par_k, t=3)


  idx_l = list(range(0,4)) + list(range(5,9))

  val = torch.cat([    node_par[0,0][None], node_par[0,0][None]*0+1, npar[0,0,:,0,idx_l].T],dim=0).numpy()
  print(val)

  '''
  array([[6, 6, 7, 0, 0, 0, 7, 5, 0, 0],
         [8, 6, 7, 8, 0, 0, 7, 5, 6, 0],
         [6, 8, 7, 8, 0, 0, 7, 5, 6, 0],
         [6, 6, 8, 8, 0, 0, 7, 5, 7, 0],
         [6, 6, 7, 8, 0, 0, 7, 5, 0, 0]])
  
  '''  
  expected  = array([
       [6, 6, 7, 0, 0, 0, 7, 5, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [8, 6, 7, 8, 0, 0, 7, 5, 6, 0],
       [6, 8, 7, 8, 0, 0, 7, 5, 6, 0],
       [6, 6, 8, 8, 0, 0, 7, 5, 7, 0],
       [6, 6, 7, 8, 0, 0, 7, 5, 0, 0],
       [6, 6, 7, 8, 0, 8, 7, 5, 0, 0],
       [6, 6, 7, 8, 0, 0, 8, 5, 7, 0],
       [6, 6, 7, 8, 0, 0, 7, 8, 5, 0],
       [6, 6, 7, 8, 0, 0, 7, 5, 0, 0]])
  assert (val == expected).all()


  val = torch.cat([    node_par_k[0,0][None], node_par_k[0,0][None]*0+1, npark[0,0,:,0,idx_l].T],dim=0).numpy()
  expected = array([
       [ 1,  6,  2, -1, -1, -1,  7,  3, -1, -1],
       [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
       [ 7,  6,  2,  0, -1, -1,  7,  3,  1, -1],
       [ 1,  7,  2,  0, -1, -1,  7,  3,  6, -1],
       [ 1,  6,  7,  0, -1, -1,  7,  3,  2, -1],
       [ 1,  6,  2,  0, -1, -1,  7,  3, -1, -1],
       [ 1,  6,  2,  0, -1,  7,  7,  3, -1, -1],
       [ 1,  6,  2,  0, -1, -1,  7,  3,  7, -1],
       [ 1,  6,  2,  0, -1, -1,  7,  7,  3, -1],
       [ 1,  6,  2,  0, -1, -1,  7,  3, -1, -1]])
  print(val)
  assert (val == expected).all()


  print(
    torch.cat([    node_par_k[0,0][None], node_par_k[0,0][None]*0+1, npark[0,0,:,1,idx_l].T],dim=0).numpy()
  )

  val = (opt_prior.exp()[0,0,:,idx_l] > 0.01).long().numpy()
  expected = array([
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0],
       [1, 1, 1, 0, 0, 1, 1, 0]])

  print(val)
  assert (val == expected).all()


  ### test fitting with synthetic data
  params = AttrDict(dict(
    L = 5,
    K = 7,
    E = 11,
    B = 3,
    M = 13,
    EPS = 1E-8,

  ))
  params.V = 20
  params.T = params.L
  device = None  
  model = DPT(params,device)
  tok_ext = torch.randint(0, params.V,size=(B,params.L))
  # print()

  loss  = model.loss_joint_lp(tok_ext)

  nstep = 200
  lr = 0.01
  optimizer = torch.optim.RMSprop(model.parameters(),lr=lr)
  # optimizer =
  data_iter = [tok_ext]
  for stepp in range(nstep):
    for dataa in data_iter:
      optimizer.zero_grad()
      loss  = model.loss_joint_lp(tok_ext)

      loss.backward()
      optimizer.step()
      print(f"[step={stepp}] loss={loss.detach().numpy():.3f}")
    
  print('[all_test_passed]')


  ### testing sampling from a fitted model
  
  breakpoint()
