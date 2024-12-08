import torch

import cv2
from model.vae import *
from model.vae_sample import *
from loss.vae_lower_bound_loss import VAELowerBoundLoss, evidence_lower_bound
from dataset.omniglot_dataset import OmniglotDataset

from torch.utils.data import DataLoader

from easydict import EasyDict
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
args = EasyDict()
args['data'] = EasyDict()
args['data']['root_dir'] = '../data'
args['data']['image_subdir'] = 'omniglot/images_background'
args['data']['stroke_subdir'] = 'omniglot/strokes_background'
args['data']['overfitting'] = False

args['network'] = EasyDict()
args['network']['EncoderNetwork'] = EasyDict()
args['network']['EncoderNetwork']['input_dimensions'] = 28 * 28
args['network']['EncoderNetwork']['hidden_layers'] = [200, 200]
args['network']['EncoderNetwork']['latent_dimensions'] = 50
args['network']['EncoderNetwork']['activation'] = "tanh"

args['network']['DecoderNetwork'] = EasyDict()
args['network']['DecoderNetwork']['output_dimension'] = 28 * 28
args['network']['DecoderNetwork']['hidden_layers'] = [200, 200]
args['network']['DecoderNetwork']['latent_dimensions'] = 50
args['network']['DecoderNetwork']['activation'] = "tanh"
args['network']['DecoderNetwork']['output_activation'] ="sigmoid"

args['train'] = EasyDict()
args['train']['batch_size'] = 20
args['train']['optimizer_params'] = EasyDict()
args['train']['optimizer_params']['betas'] = (0.9, 0.999)
args['train']['optimizer_params']['eps'] = 1e-4
args['train']['optimizer_params']['lr'] = 1e-3

train_dataset = OmniglotDataset(args['data'])
train_loader = DataLoader(train_dataset, batch_size=args.train.batch_size, shuffle=True)
print("[main]: Iterations per epoch: ", len(train_loader))

loss_func = VAELowerBoundLoss()
# loss_func = evidence_lower_bound
model = VAE(args['network'])
model = model.to('cuda')

def save_image(x, y, prefix):
    image_x = x.detach().cpu().numpy().transpose((0,2,3,1))
    image_y = y.detach().cpu().numpy().transpose((0,2,3,1))
    image_x = (image_x * 255).astype(np.uint8)
    image_y = (image_y * 255).astype(np.uint8)
    for i in range(y.shape[0]):
        cv2.imwrite('../logs/{}_{:03d}_image_y.png'.format(prefix, i), image_y[i])
        cv2.imwrite('../logs/{}_{:03d}_image_x.png'.format(prefix, i), image_x[i])
        k = cv2.waitKey()
        
def show_image(x, y):
    image_x = x.detach().cpu().numpy().transpose((0,2,3,1))
    image_y = y.detach().cpu().numpy().transpose((0,2,3,1))
    image_x = (image_x * 255).astype(np.uint8)
    image_y = (image_y * 255).astype(np.uint8)
    for i in range(y.shape[0]):
        cv2.imshow('image_y', image_y[i])
        cv2.imshow('image_x', image_x[i])
        k = cv2.waitKey()
        
optimizer = Adam(model.parameters(), **args.train.optimizer_params)
EPOCH = 3280
I = 7
total_iterations = sum([3**i for i in range(I+1)]) * len(train_loader)
accum_iter = 0

with tqdm(total=total_iterations) as pbar:
    for i in range(I+1):
        lr_scaler = 10 ** (-i/7.0)
        new_lr = args['train']['optimizer_params']['lr'] * lr_scaler

        # -- lr scheduling --
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        
        num_iterations = 3 ** i
        
        for sub_iteration in range(num_iterations):
            
            for it, sample in enumerate(train_loader):
                for k, v in sample.items():
                    sample[k] = v.cuda()
                
                B, C, H, W = sample['image'].shape
                net_in = sample['image'].flatten(start_dim=1)
                
                optimizer.zero_grad()
                
                mean, logvar, z, y = model(net_in)
                y = y.view(B, C, H, W)
                likelihood_params = {'theta': y[:,None,...]}
                input = {
                        'log_var': logvar,
                        'mu': mean,
                        'x': sample['image'],
                        'likelihood_params': likelihood_params
                        }
                
                loss = loss_func(input)
                
                loss.backward()
                optimizer.step()
                
                if accum_iter % 30000 == 0:
                    save_image(sample['image'], y, prefix="{:07d}".format(accum_iter)) 
                    
                pbar.update(1)
                accum_iter += 1
                if it % 100 == 0:
                    pbar.set_description("Loss {:.2f}, lr: {:.5f}".format(loss.item(), new_lr))
                

# save_image(sample['image'], x_hat) 

