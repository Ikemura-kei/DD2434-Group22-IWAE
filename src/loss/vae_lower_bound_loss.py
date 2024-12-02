import torch
from torch.nn import Module
import numpy

class VAELowerBoundLoss(Module):
    def __init__(self):
        self.likelihood_type = 'bernoulli'
    
    def forward(self, input):
        exp_sigma = input['exp_sigma']
        mu = input['mu']
        eps = input['eps'] 
        x = input['x']
        likelihood_params = input['likelihood_params']
        
        return self._compute(exp_sigma, mu, eps, x, likelihood_params)
    
    def _compute(self, exp_sigma, mu, eps, x, likelihood_params):
        sigma = torch.log(exp_sigma)
        term1 = 0.5 * (1 +torch.log(sigma)-sigma-torch.square(mu))
        
        z = eps * sigma + mu
        term2 = torch.mean(self._log_likelihood(x, likelihood_params), dim=-1)
        
        return torch.mean(term1 + term2)
    
    def _log_likelihood(self, x, likelihood_params):
        if self.likelihood_type == 'bernoulli':
            return x * torch.log(likelihood_params['theta']) + (1-x) * torch.log(1-likelihood_params['theta'])
        else:
            raise NotImplementedError