# -- model import --
from model.vae import *

# -- loss import --
from loss import *

# -- dataset import --
from dataset.omniglot_dataset import OmniglotDataset
from dataset.omniglot_original_dataset import OmniglotOriginalDataset

# -- torch stuff --
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from easydict import EasyDict
from tqdm import tqdm
from torch.optim import Adam
import numpy as np

# -- utils import --
from utils.eval import evaluation
from utils.misc import dict2edict, save_images

# -- other stuff --
import yaml
import argparse
import os
from datetime import datetime
import shutil

# -- parse external arguments --
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./cfg/sample.yml')
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--ckpt', type=str, default='')
args = parser.parse_args()

# -- read configuration --
with open(args.cfg, 'r') as file:
    cfg_dict = yaml.safe_load(file)
cfg = dict2edict(cfg_dict)

if not args.eval:
    # -- create log directory and save configuration used --
    save_dir = os.path.join('./logs', datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs(save_dir, exist_ok=True)
    import json
    with open(os.path.join(save_dir, '{}.json'.format(args.cfg.split('/')[-1].split('.')[0])), 'w') as f:
        json.dump(cfg, f, indent=4)
else:
    assert os.path.exists(args.ckpt)
    save_dir = os.path.join(*args.ckpt.split('/')[:-1], 'eval')
    if args.ckpt[0] == '/':
        save_dir = '/' + save_dir
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    cfg['ckpt_path'] = args.ckpt
    import json
    with open(os.path.join(save_dir, '{}.json'.format(args.cfg.split('/')[-1].split('.')[0])), 'w') as f:
        json.dump(cfg, f, indent=4)
    
# -- create a logger file --
log_file = open(os.path.join(save_dir, 'logs.txt'), 'w')
log_file.close()

# -- copy the current version of code --
shutil.copyfile(os.path.abspath(__file__), os.path.join(save_dir, 'code.py'))

# -- dataset --
if cfg.data.name == 'Omniglot':
    train_dataset = OmniglotOriginalDataset(cfg['data'], train=True)
    test_dataset = OmniglotOriginalDataset(cfg['data'], train=False)
elif cfg.data.name == 'MNIST':
    train_dataset = datasets.MNIST(root=cfg['data'].root_dir, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root=cfg['data'].root_dir, train=False, transform=transforms.ToTensor(), download=True) 
else:
    raise NotImplementedError

train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=10)
test_loader = DataLoader(test_dataset, batch_size=35, shuffle=False, num_workers=10)
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
if args.eval:
    model.load_state_dict(torch.load(args.ckpt))
model = model.to('cuda')

# -- optimizer --
optimizer = Adam(model.parameters(), **cfg.train.optimizer_params)

# -- start training --
I = 7
total_iterations = sum([3**i for i in range(I+1)]) * len(train_loader)
if not args.eval:
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
                        nll, image_in, image_out = evaluation(model, test_loader, save_dir, eval_metric, accum_iter)
                        
                        image_out = image_out[:,0,...]
                        image_out = image_out.view(-1, 1, 28, 28)
                        image_in = image_in.view(-1, 1, 28, 28)
                        save_sub_dir = os.path.join(save_dir, 'iter_{:07d}'.format(accum_iter))
                        os.makedirs(save_sub_dir, exist_ok=True)
                        save_images(image_out, save_dir=save_sub_dir, suffix='pred')
                        save_images(image_in, save_dir=save_sub_dir, suffix='gt')
                        
                        model.train()
                        log_file = open(os.path.join(save_dir, 'logs.txt'), "a")
                        log_file.write("iteration {:07d}, NLL {:.2f}, lr: {:.5f}\n".format(accum_iter, nll, new_lr))
                        log_file.close()

nll, _, _ = evaluation(model, test_loader, save_dir, eval_metric, total_iterations-1)
log_file = open(os.path.join(save_dir, 'logs.txt'), "a")
log_file.write("Final, NLL {:.2f}".format(nll))
log_file.close()