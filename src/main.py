import torch

import cv2
from model.vae import *
from loss.vae_lower_bound_loss import VAELowerBoundLoss, evidence_lower_bound
from dataset.omniglot_dataset import OmniglotDataset
from dataset.omniglot_original_dataset import OmniglotOriginalDataset

from torch.utils.data import DataLoader

from easydict import EasyDict
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
from iwae_model import *
from iwae_loss import *


args = EasyDict()
args['data'] = EasyDict()
args['data']['root_dir'] = './data'
args['data']['image_subdir'] = ['omniglot/images_background', 'omniglot/images_background_small1', 'omniglot/images_background_small2', 'omniglot/images_evaluation']
args['data']['stroke_subdir'] = ['omniglot/strokes_background', 'omniglot/strokes_background_small1', 'omniglot/strokes_background_small2', 'omniglot/strokes_evaluation']
# args['data']['image_subdir'] = ['omniglot/images_background', 'omniglot/images_background_small2']
# args['data']['stroke_subdir'] = ['omniglot/strokes_background', 'omniglot/strokes_background_small2']
args['data']['overfitting'] = False

args['network'] = EasyDict()
args['network']['EncoderNetwork'] = EasyDict()
args['network']['EncoderNetwork']['input_dimensions'] = 28 * 28
args['network']['EncoderNetwork']['hidden_layers'] = [200, 200]
args['network']['EncoderNetwork']['latent_dimensions'] = 50
args['network']['EncoderNetwork']['activation'] = "tanh"
args['network']['EncoderNetwork']['k'] = 1

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

import os
from datetime import datetime
save_dir = os.path.join('./logs', datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
os.makedirs(save_dir, exist_ok=True)
import json
with open(os.path.join(save_dir, 'cfg.json'), 'w') as f:
    json.dump(args, f, indent=4)
    
# train_dataset = OmniglotDataset(args['data'])
train_dataset = OmniglotOriginalDataset(args['data'], train=True)
test_dataset = OmniglotOriginalDataset(args['data'], train=False)
train_loader = DataLoader(train_dataset, batch_size=args.train.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("[main]: Iterations per epoch: ", len(train_loader))

loss_func = VAELowerBoundLoss()
eval_metric = IWAELowerBoundLoss()
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
                
                
                B, C, HW = sample['image'].shape
                # net_in = sample['image'].flatten(start_dim=1)
                net_in = sample['image'].view(-1, 1, 28 * 28)
                optimizer.zero_grad()
                
                mean, logvar, z, y = model(net_in)
                # print(y.shape)
                # y = y.view(B, C, H, W)
                # likelihood_params = {'theta': y}
                # input = {
                #         'log_var': logvar,
                #         'mu': mean,
                #         'x': net_in,
                #         'likelihood_params': likelihood_params
                #         }
                # loss = loss_func(input)
                
                input = {
                            'log_var': logvar,
                            'mu': mean,
                            'x': net_in,
                            'recon_x': y,
                            'repar_z': z,
                            'k': args['network']['EncoderNetwork']['k']
                            }
                loss = eval_metric(input)
                
                
                loss.backward()
                optimizer.step()
                
                    
                pbar.update(1)
                accum_iter += 1
                if it % 100 == 0:
                    pbar.set_description("Loss {:.2f}, lr: {:.5f}".format(loss.item(), new_lr))
                
                if accum_iter % (39950 * 2) == 0:
                    # save_image(sample['image'], y, prefix="{:07d}".format(accum_iter)) 
                    
                    model.eval()
                    # train_dataset.eval()
                    
                    model.encoder.k = 5000
                    with torch.no_grad():
                        eval_loss = 0
                        n_samples = 0
                        for _, sample in enumerate(test_loader):
                            for k, v in sample.items():
                                sample[k] = v.cuda()
                            
                            # B, C, HW = image.shape
                            # flattened_img = sample['image'].flatten(start_dim=1)
                            flattened_img = sample['image'].view(-1, 1, 28 * 28)
                            mean, logvar, z, y = model(flattened_img)

                            input = {
                                        'log_var': logvar,
                                        'mu': mean,
                                        'x': flattened_img,
                                        'recon_x': y,
                                        'repar_z': z,
                                        'k': 5000
                                        }
                            loss = eval_metric(input)
                            eval_loss += (loss * sample['image'].shape[0])
                            n_samples += sample['image'].shape[0]
                            # print(loss)

                        eval_loss = eval_loss / n_samples
                        print("eval loss", eval_loss)
                        torch.save(model.state_dict(), os.path.join(save_dir, "iwae_model_iter{:07d}_nll{:.3f}.pth".format(accum_iter, eval_loss.item())))
                    model.encoder.k = args['network']['EncoderNetwork']['k']
                    model.train()

# save_image(sample['image'], x_hat) 

