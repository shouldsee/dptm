
import torch.nn as nn
import torch

import math
class VAEL1(nn.Module):
  '''
  '''
  def __init__(self, 
      params,
      device=None,
      ):
      super().__init__()

      # self.d_model = d_model
      # self.dropout_emb  = nn.Dropout(dropout)
      self.params = params



      V = params.V
      if not isinstance(V,int):
        V = math.prod(*V)
      # self.params = AttrDict()
      if device is None:
        device = torch.device('cpu')
      self.device = device

      E = params.E

      
      self.w_enc = nn.Linear(V,E).to(device)
      self.w_dec = nn.Linear(E,V).to(device)
      self.b_scale_out = torch.tensor(0.,device=device)
      self.b_loc_prior = torch.tensor(0.,device=device)
      self.b_scale_prior = torch.tensor(0.,device=device)





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
    pass

  def forward(self, tok_emb, h0=None):
    B,L,E = tok_emb.shape
    if h0 is None:
        h0 = 0* tok_emb[:,0:1,:].expand(B,self.params.D,E).transpose(0,1) 

    output_emb, hn = self.rnn(tok_emb, h0 )
    return output_emb
    


  def loss_joint_lp(self, dat):
    img,lab = dat
    w_enc = self.w_enc
    w_dec = self.w_dec
    b_scale_out = self.b_scale_out
    b_loc_prior = self.b_loc_prior
    b_scale_prior = self.b_scale_prior


    img_flat = img.reshape((len(img),-1))
    B, L = img.shape
    E = self.params.E
    D = self.params.D

    scale = self.params.betasq_post
    NS = self.params.inner_sample
    loc = w_enc(img_flat)
    prior = torch.distributions.normal.Normal(b_loc_prior, b_scale_prior.exp())
    post = torch.distributions.normal.Normal(loc, scale.exp())
    z_noise = post.rsample((B,E,NS)).transpose(1,2)


    scale = self.params.betasq_prior
    # scale = self.params.betasq_e
    loc = w_dec(z_noise)
    x_output = torch.distributions.normal.Normal(loc=loc,scale=b_scale_out.exp())
    ### (B,NS,L)
    lp_external = x_output.log_prob(img_flat)
    lp_internal = prior.log_prob(z_noise) - post.log_prob(z_noise)

    ### estimate partition function
    ### (B,)
    lse = (lp_internal+lp_external).logsumexp(1) - math.log(NS)




    loss = - lp_joint_bm.mean(0)


    # loss = -( lp_joint_bm.logsumexp(-1).sum(0))
    return loss
