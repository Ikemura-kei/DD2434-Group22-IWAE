import torch
from torch.nn import Module
import numpy

class IWAELowerBoundLoss(Module):
    def __init__(self):
        pass
    
    def _compute(self):
        # -- get log(p(z)), the prior --
        
        # -- get log(p(x|z)), the likelihood, using decoder --
        
        # -- get log(q(z|x)), the approximate posterior, using the encoder --
        
        
        # -- compute log weight, that is log(w) = log(p(x,z)/q(z|x)) = log(p(z)) + log(p(x|z)) - log(q(z|x)) --
        
        # -- remove the max from the log weights --
        
        # -- compute the weights, that is w = exp{log(w)} --
        
        # -- compute the normalized weights --
        
        
        
        pass