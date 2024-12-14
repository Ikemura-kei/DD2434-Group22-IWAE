import torch
from torch.nn import Module
import numpy

class VAELowerBoundLoss(Module):
    def __init__(self):
        super().__init__()
        self.likelihood_type = 'bernoulli'
    
    def forward(self, input):
        log_var = input['log_var']
        mu = input['mu']
        x = input['x']
        likelihood_params = input['likelihood_params']
        
        return self._compute(log_var, mu, x, likelihood_params)
    
    def _compute(self, log_var, mu, x, likelihood_params):
        # log_var: exp(sigma^2) (B, J), J being the dimension of the multi-variate Gaussian
        # mu: (B, J)
        # x: inputs, (B, C, H, W), C=1 if Bernoulli likelihood, C>1 otherwise
        # x: inputs, (B, 1, C*H*W), C=1 if Bernoulli likelihood, C>1 otherwise
        var = torch.exp(log_var)
        term1 = -0.5 * (1 + log_var - var - torch.square(mu)).sum()
        
        log_likelihood = self._log_likelihood(x, likelihood_params)
        reconstruction_losses_per_input = torch.sum(torch.flatten(log_likelihood, start_dim=1), dim=1)
        term2 = -1 * torch.mean(reconstruction_losses_per_input) # we want to minimize loss, so multiply -1
        
        # print("Reconstruction losses {}".format(reconstruction_losses_per_input))
        # print("D_KL loss {}".format(term1))
        # print("Reconstruction loss mean {}".format(term2))
        
        return term1 + term2
    
    def _log_likelihood(self, x, likelihood_params):
        if self.likelihood_type == 'bernoulli':
            # likelihood_params['theta']: (B, K, H*W)
            K = likelihood_params['theta'].shape[1]
            x_ = torch.tile(x, (1, K, 1))
            return x_ * torch.clamp(torch.log(likelihood_params['theta'] + 1e-7), min=-100, max=1e6) + (1-x_) * torch.clamp(torch.log(1-likelihood_params['theta'] + 1e-7), min=-100, max=1e6)
        else:
            raise NotImplementedError
        
import torch
import torch.nn as nn
import torch.nn.functional as F


def evidence_lower_bound(input):
    """
    ELBO for Bernoulli VAE
    """
    log_var = input['log_var']
    mean = input['mu']
    x = input['x']
    x_hat = input['likelihood_params']['theta']
    # x, x_hat, mean, log_var
    reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
    DKL = 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    # print(DKL, reconstruction_loss)
    return reconstruction_loss - DKL
        
if __name__ == "__main__":
    # run some basic tests to make sure the computation works
    vae_loss = VAELowerBoundLoss()
    
    B = 10
    H = W = 4
    C = 1
    K = 1
    J = 10
    
    vars = torch.ones((B, J))
    mu = torch.rand((B, J))
    x = torch.rand((B, C, H, W))
    x = torch.where(x > 0.5, 1., 0.)
    
    y = torch.zeros((B, K, C, H, W))
    for k in range(K):
        y[:,k,...] = torch.where(x < 1., x + 1e-5, x - 1e-5)
    # -- reconstruction loss at index 2 should be zero --
    y[2][0] = x[2]
    # -- reconstruction loss at index 5 should be high --
    y[5][0] = torch.where(x[5] < 1., x[5] + 3e-2, x[5] - 3e-2) 
    likelihood_params = {
        'theta': y
    }
    
    input = {
            'log_var': torch.log(vars),
            'mu': mu,
            'x': x,
            'likelihood_params': likelihood_params
            }
    
    vae_loss(input)