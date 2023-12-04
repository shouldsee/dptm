
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

      self.params = params

      V = params.V
      if not isinstance(V,int):
        V = math.prod(V)
      FV = V  ### flat V

      if device is None:
        device = torch.device('cpu')
      self.device = device

      E = params.E

      
      self.w_enc = nn.Linear(FV,E).to(device)
      self.w_dec = nn.Linear(E,FV).to(device)
      self.b_scale_out = nn.Parameter(torch.tensor(0.,device=device))
      self.b_loc_prior = nn.Parameter(torch.tensor(0.,device=device))
      self.b_scale_prior = nn.Parameter(torch.tensor(0.,device=device))
      self.b_scale_post = nn.Parameter(torch.tensor(0.,device=device))
      


  def sample(self, 
    shape, tok):
    shape = tuple(shape)

    w_enc = self.w_enc
    w_dec = self.w_dec
    b_scale_out = self.b_scale_out
    b_scale_post = self.b_scale_post
    b_loc_prior = self.b_loc_prior
    b_scale_prior = self.b_scale_prior    
    prior = torch.distributions.normal.Normal(b_loc_prior, b_scale_prior.exp())
    z_noise = prior.sample(shape+(self.params.  E,))



    loc = w_dec(z_noise)
    loc = loc.reshape(shape+tuple(self.params.V))
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
    w_enc = self.w_enc
    w_dec = self.w_dec
    b_scale_out = self.b_scale_out
    b_scale_post = self.b_scale_post
    b_loc_prior = self.b_loc_prior
    b_scale_prior = self.b_scale_prior


    img_flat = img.reshape((len(img),-1))
    # B, L = img.shape
    B = img.size()[0]
    E = self.params.E

    NS = self.params.inner_nsample
    loc = w_enc(img_flat)
    prior = torch.distributions.normal.Normal(b_loc_prior, b_scale_prior.exp())
    post = torch.distributions.normal.Normal(loc, b_scale_post.exp())
    z_noise = post.rsample((NS,))



    # scale = self.params.betasq_prior
    # scale = self.params.betasq_e
    loc = w_dec(z_noise)
    x_output = torch.distributions.normal.Normal(loc=loc,scale=b_scale_out.exp())
    ### (NS,B,L)
    lp_external = x_output.log_prob(img_flat)
    lp_internal = prior.log_prob(z_noise) - post.log_prob(z_noise)

    ### estimate partition function
    ### (B,)
    lse = (lp_internal.sum(-1)+lp_external.sum(-1)).logsumexp(0) - math.log(NS)




    loss = - lse.mean(0)


    # loss = -( lp_joint_bm.logsumexp(-1).sum(0))
    return loss
