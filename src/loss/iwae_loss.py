import numpy as np
import torch
import torch.distributions as dists
from easydict import EasyDict 
from torch import nn



class IWAELowerBoundLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        log_var = input['log_var']
        mu = input['mu']
        x = input['x']
        recon_x = input['recon_x']
        repar_z = input['repar_z']
        k = input['k']
        return self._compute(log_var, mu, x, recon_x, repar_z, k)   
        
    def _compute(self, log_var, mu, x, recon_x, repar_z, k):
        x = (x > 0.5).float()  # Binarizing input

        # Collect K samples
        # mu_sampled = mu.unsqueeze(1).repeat(1, k, 1) 
        mu_sampled = mu
        # sigma_sampled = (0.5 * log_var).exp().unsqueeze(1).repeat(1, k, 1)
        sigma_sampled = (0.5 * log_var).exp()
        # x_sampled = x.unsqueeze(1).repeat(1, k, 1) 
        x_sampled = x
        # recon_x = recon_x.unsqueeze(1).repeat(1, k, 1)
        # repar_z = repar_z.unsqueeze(1).repeat(1, k, 1)
        # print(recon_x.shape)
        # Compute unnormalized log weights [batch, sample_num]
        log_p_x_given_z = dists.Bernoulli(recon_x).log_prob(x_sampled).sum(2)
        log_p_z = dists.Normal(0, 1).log_prob(repar_z).sum(2)
        log_q_z_given_x = dists.Normal(mu_sampled, sigma_sampled).log_prob(repar_z).sum(2)
        log_importance_weight = log_p_x_given_z - log_q_z_given_x + log_p_z # (B, K)

        # Compute IWAE loss [batch] -> scalar
        log_marginal_likelihood = (torch.logsumexp(log_importance_weight, 1) - np.log(k)).mean() # mean over batch
        iwae_loss = - log_marginal_likelihood

        return iwae_loss


