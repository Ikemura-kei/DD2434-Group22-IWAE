import torch
from torch.nn import Module
import numpy

class VAELowerBoundLoss(Module):
    def __init__(self):
        pass
    
    def forward(self, input):
        exp_sigma = input['exp_sigma']
        mu = input['mu']
        eps = input['eps'] 
        x = input['x']
        
        return self._compute(exp_sigma, mu, eps, x)
    
    def _compute(self, exp_sigma, mu, eps, x):
        sigma = torch.log(exp_sigma)
        term1 = 0.5 * (1 +torch.log(sigma)-sigma-torch.square(mu))
        
        z = eps * sigma + mu
        term2 = torch.mean(self._likelihood(x, z), dim=-1)
        
        return torch.mean(term1 + term2)