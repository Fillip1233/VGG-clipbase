import torch
def f(src):
    ###src(475,9,256)
    src_zero=torch.zeros(src.shape[0]*2,src.shape[1],src.shape[2]).shape
    for b in range(src.shape[1]-1):
        src_zero[:,b,:] = torch.cat(src[:,b,:],src[:,b+1,:], dim = 0)
    src_zero[:,src.shape[1],:]=src_zero[:,src.shape[1]-1,:]
        
    