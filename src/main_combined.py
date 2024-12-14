import torch
import os
from datetime import datetime

import cv2
from model.vae import *
from loss.vae_lower_bound_loss import VAELowerBoundLoss, evidence_lower_bound
from dataset.omniglot_dataset import OmniglotDataset
from dataset.omniglot_original_dataset import OmniglotOriginalDataset

from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from easydict import EasyDict
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
from iwae_model import *
from iwae_loss import *
from utils.eval import evaluation
from utils.misc import dict2edict
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./cfg/sample.yml')
args = parser.parse_args()

# -- read configuration --
with open(args.cfg, 'r') as file:
    cfg_dict = yaml.safe_load(file)
cfg = dict2edict(cfg_dict)

# -- create log directory and save configuration used --
save_dir = os.path.join('./logs', datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
os.makedirs(save_dir, exist_ok=True)
import json
with open(os.path.join(save_dir, 'cfg.json'), 'w') as f:
    json.dump(cfg, f, indent=4)

# -- dataset --
if cfg.data.name == 'Omniglot':
    train_dataset = OmniglotOriginalDataset(cfg['data'], train=True)
    test_dataset = OmniglotOriginalDataset(cfg['data'], train=False)
elif cfg.data.name == 'MNIST':
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True) 
else:
    raise NotImplementedError

train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("[main]: Iterations per epoch: ", len(train_loader))

# -- loss --
if cfg.loss.name == 'VAE':
    loss_func = VAELowerBoundLoss()
elif cfg.loss.name == 'IWAE':
    loss_func = IWAELowerBoundLoss()
else:
    raise NotImplementedError
eval_metric = IWAELowerBoundLoss()

# -- model --
model = VAE(cfg['network'])
model = model.to('cuda')

# -- optimizer --
optimizer = Adam(model.parameters(), **cfg.train.optimizer_params)

# -- start training --
I = 7
total_iterations = sum([3**i for i in range(I+1)]) * len(train_loader)
accum_iter = 0

with tqdm(total=total_iterations) as pbar:
    for i in range(I+1):
        lr_scaler = 10 ** (-i/7.0)
        new_lr = cfg['train']['optimizer_params']['lr'] * lr_scaler

        # -- lr scheduling --
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        
        num_iterations = 3 ** i
        
        for sub_iteration in range(num_iterations):
            
            for it, sample in enumerate(train_loader):
                image = sample[0]
                image = image.to('cuda')
                
                net_in = image.view(-1, 1, 28 * 28)
                optimizer.zero_grad()
                
                mean, logvar, z, y = model(net_in)
                
                if cfg.loss.name == 'VAE':
                    likelihood_params = {'theta': y}
                    input = {
                            'log_var': logvar,
                            'mu': mean,
                            'x': net_in,
                            'likelihood_params': likelihood_params
                            }
                elif cfg.loss.name == 'IWAE':
                    input = {
                            'log_var': logvar,
                            'mu': mean,
                            'x': net_in,
                            'recon_x': y,
                            'repar_z': z,
                            'k': model.encoder.k
                            }
                    
                loss = loss_func(input)
                
                loss.backward()
                optimizer.step()
                
                pbar.update(1)
                accum_iter += 1
                if it % 100 == 0:
                    pbar.set_description("Loss {:.2f}, lr: {:.5f}".format(loss.item(), new_lr))
                
                if accum_iter % cfg.train.eval_period == 0:
                    evaluation(model, test_loader, save_dir, eval_metric, accum_iter)
                    model.train()
    

