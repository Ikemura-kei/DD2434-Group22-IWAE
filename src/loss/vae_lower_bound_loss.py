import torch
from torch.nn import Module
import numpy
import torch.distributions as dists

class VAELowerBoundLoss(Module):
    def __init__(self, likelihood_type='bernoulli'):
        super().__init__()
        self.likelihood_type = likelihood_type
    
    def forward(self, input):
        log_var = input['log_var']
        mu = input['mu']
        x = input['x']
        likelihood_params = input['likelihood_params']
        
        return self._compute(log_var, mu, x, likelihood_params)
    
    def _compute(self, log_var, mu, x, likelihood_params):
        # log_var: exp(sigma^2) (B, J), J being the dimension of the multi-variate Gaussian
        # mu: (B, J)
        # x: inputs, (B, 1, C*H*W)
        x = (x > 0.5).float()  # Binarizing input
        var = torch.exp(log_var)
        term1 = -0.5 * (1 + log_var - var - torch.square(mu)).sum(dim=-1).squeeze(-1)
        
        log_likelihood = self._log_likelihood(x, likelihood_params)
        reconstruction_losses_per_input = torch.sum(log_likelihood, dim=-1)
        term2 = -1 * torch.mean(reconstruction_losses_per_input, dim=-1) # we want to minimize loss, so multiply -1
        
        loss = (term1 + term2)
        loss = loss.mean()
        return loss
    
    def _log_likelihood(self, x, likelihood_params):
        if self.likelihood_type == 'bernoulli':
            # likelihood_params['theta']: (B, K, H*W)
            return dists.Bernoulli(likelihood_params['theta']).log_prob(x)
            # K = likelihood_params['theta'].shape[1]
            # x_ = torch.tile(x, (1, K, 1))
            # return x_ * torch.clamp(torch.log(likelihood_params['theta'] + 1e-100), min=-1000, max=1e6) + (1-x_) * torch.clamp(torch.log(1-likelihood_params['theta'] + 1e-100), min=-1000, max=1e6)
        else:
            raise NotImplementedError
        
if __name__ == "__main__":
    pass