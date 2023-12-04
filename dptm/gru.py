
import torch.nn as nn
import torch
class GRUWrapper(nn.Module):
  '''
  Dynamic probabilistic tree models
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
      # self.params = AttrDict()
      if device is None:
        device = torch.device('cpu')
      self.device = device

      params.D = D =  3

      self.rnn = nn.GRU(E, E, D, batch_first=True,).to(device)

      self.emb_vocab  = nn.Parameter(nn.Linear(E, V,).weight.to(device))


  def _inner_loop(self, bm_dict_next, t, max_t, is_ar_loss):
      pass



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
    B, L = tok_external.shape
    E = self.params.E
    # B,L,E = tok_emb.shape
    D = self.params.D

    h0 = torch.zeros((B,D,E),device=self.device).transpose(0,1).contiguous()
    tok_input = torch.cat([tok_external[:,:1]*0,tok_external[:,:-1]],dim=1).contiguous()
    output_logit, hn = self.forward(tok_input, h0)

    lp_joint_bm = torch.gather( output_logit, index=tok_external.unsqueeze(-1),dim=-1).squeeze()
    lp_joint_bm = lp_joint_bm.sum(dim=1)
    loss = - lp_joint_bm.mean(0)


    # loss = -( lp_joint_bm.logsumexp(-1).sum(0))
    return loss



class GRUWrapperGaussian(nn.Module):
  '''
  Dynamic probabilistic tree models
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

    #   V = params.V
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


  def forward(self, tok_emb, h0=None):
    B,L,E = tok_emb.shape
    if h0 is None:
        h0 = 0* tok_emb[:,0:1,:].expand(B,self.params.D,E).transpose(0,1) 

    output_emb, hn = self.rnn(tok_emb, h0 )
    return output_emb
    


  def loss_joint_lp(self, dat):
    img,lab = dat

    img = img.reshape((len(img),-1))
    B, L = img.shape
    E = self.params.E
    D = self.params.D
    input = img[:,:-1]
    output = img[:,1:]
    h0 = 0* input[:,0:1,:].expand(B,self.params.D,E).transpose(0,1) 



    output_emb, hn = self.rnn(tok_emb, h0 )

    h0 = torch.zeros((B,D,E),device=self.device).transpose(0,1).contiguous()
    tok_input = torch.cat([tok_external[:,:1]*0,tok_external[:,:-1]],dim=1).contiguous()
    output_logit, hn = self.forward(tok_input, h0)

    lp_joint_bm = torch.gather( output_logit, index=tok_external.unsqueeze(-1),dim=-1).squeeze()
    lp_joint_bm = lp_joint_bm.sum(dim=1)
    loss = - lp_joint_bm.mean(0)


    # loss = -( lp_joint_bm.logsumexp(-1).sum(0))
    return loss






class PCAGaussian(nn.Module):
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
      # is_ar_loss       = params.is_ar_loss
      # is_ar_loss = False

    #   V = params.V
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


  def forward(self, tok_emb, h0=None):
    B,L,E = tok_emb.shape
    if h0 is None:
        h0 = 0* tok_emb[:,0:1,:].expand(B,self.params.D,E).transpose(0,1) 

    output_emb, hn = self.rnn(tok_emb, h0 )
    return output_emb
    


  def loss_joint_lp(self, dat):
    raise NotImplementedError()
    img,lab = dat

    img = img.reshape((len(img),-1))
    B, L = img.shape
    E = self.params.E
    D = self.params.D
    input = img[:,:-1]
    output = img[:,1:]
    h0 = 0* input[:,0:1,:].expand(B,self.params.D,E).transpose(0,1) 



    output_emb, hn = self.rnn(tok_emb, h0 )

    h0 = torch.zeros((B,D,E),device=self.device).transpose(0,1).contiguous()
    tok_input = torch.cat([tok_external[:,:1]*0,tok_external[:,:-1]],dim=1).contiguous()
    output_logit, hn = self.forward(tok_input, h0)

    lp_joint_bm = torch.gather( output_logit, index=tok_external.unsqueeze(-1),dim=-1).squeeze()
    lp_joint_bm = lp_joint_bm.sum(dim=1)
    loss = - lp_joint_bm.mean(0)


    # loss = -( lp_joint_bm.logsumexp(-1).sum(0))
    return loss
