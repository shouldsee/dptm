
import torch.nn as nn
import torch

import math
from attrdict import AttrDict
class SRBMGL2(nn.Module):
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
      Z = E+1
      params.Z = Z

      
    #   self.w_enc = nn.Linear(FV,E).to(device)
      self.w_dec = nn.Linear(E,FV).to(device)
      self.whz = nn.Linear(Z,E).to(device)
      self.b_scale_out = nn.Parameter(torch.tensor(0.,device=device))
      self.b_loc_prior = nn.Parameter( nn.Linear(Z,1).weight.reshape((Z,)).to(device))
      self.b_scale_prior = nn.Parameter(torch.tensor(0.,device=device))
      self.b_scale_post = nn.Parameter(torch.tensor(0.,device=device))
      


  def sample(self, 
    shape, tok):
    shape = tuple(shape)


    # w_enc = self.w_enc
    w_dec = self.w_dec
    b_scale_out = self.b_scale_out
    b_scale_post = self.b_scale_post
    b_loc_prior = self.b_loc_prior
    b_scale_prior = self.b_scale_prior    
    self.b_loc_prior
    

    prior = torch.distributions.Bernoulli(logits=self.b_loc_prior)
    ### (NS,B,E)

    z_noise = prior.sample(shape)
    h_post_log = self.whz(z_noise)
    h_noise = torch.distributions.Bernoulli(logits=h_post_log).sample()
    loc = w_dec(h_noise)

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
    # w_enc = self.w_enc
    w_dec = self.w_dec
    b_scale_out = self.b_scale_out
    b_scale_post = self.b_scale_post
    b_loc_prior = self.b_loc_prior
    b_scale_prior = self.b_scale_prior


    params = self.params

    img_flat = img.reshape((len(img),-1))
    # B, L = img.shape
    node_bx = img_flat

    B = img.size()[0]
    E = self.params.E
    Z = self.params.Z
    EPS = params.EPS
    # xd = self.get_init_dict(dat)
    device = self.device
    xd = AttrDict(
      node_h = torch.zeros((B,E),device=device)+EPS,
      node_z = torch.zeros((B,Z),device=device)+EPS,

    )
    node_h = xd.node_h
    node_z = xd.node_z

    inner_nstep = params.inner_nstep

    def arr_random_update(arr,arr_next, p,):
        sel = (torch.rand_like(arr)<p)
        # out = (~sel) *arr + sel*arr_next
        out = (~sel) *arr + sel*(arr_next+arr)*0.5
        return out

    inner_lr = params.inner_lr    
    p_h_update = inner_lr
    p_z_update = inner_lr

    # lidx = list(lidx)
    
    for i in range(inner_nstep):


        ### (B,L,E)
        node_e = self.w_dec(node_h)
        dh1 = torch.tensordot((node_bx - node_e ) * (b_scale_out*2).exp(), self.w_dec.weight, 1)

        node_h_post = self.whz(node_z)
        dh2 = node_h_post.sigmoid().log()
        dz1 = torch.tensordot( node_h*(1-node_h_post.sigmoid()), self.whz.weight, 1) + self.b_loc_prior.sigmoid().log()

        node_h = arr_random_update(node_h, (dh1+dh2).sigmoid(), p_h_update)    
        node_z = arr_random_update(node_z, (dz1).sigmoid(), p_z_update)    

        node_h = node_h.detach()
        node_z = node_z.detach()



    NS = self.params.inner_nsample

    post = torch.distributions.Bernoulli(node_z)
    # prior = torch.distributions.Bernoulli(torch.tensor(0.5,device=device))
    prior = torch.distributions.Bernoulli(logits=self.b_loc_prior)

    ### (NS,B,E)
    z_noise = post.sample((NS,))
    h_post_log = self.whz(z_noise)
    h_noise = torch.distributions.Bernoulli(logits=h_post_log).sample()
    loc = w_dec(h_noise)
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

