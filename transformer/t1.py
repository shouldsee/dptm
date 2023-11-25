

node_ie_exp        # (B,  M,  2L,  2K,  2L,  E)
    = node_ie.expand(1,1,1,2K,2L,1).clone()
