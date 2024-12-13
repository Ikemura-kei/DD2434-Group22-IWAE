import torch

import cv2
from model.vae import *
from model.vae_sample import *
from loss.vae_lower_bound_loss import VAELowerBoundLoss, evidence_lower_bound
from dataset.omniglot_dataset import OmniglotDataset

from torch.utils.data import DataLoader

from easydict import EasyDict

from torch.optim import Adam
import numpy as np
args = EasyDict()
args['data'] = EasyDict()
args['data']['root_dir'] = '../data'
args['data']['image_subdir'] = 'omniglot/images_background'
args['data']['stroke_subdir'] = 'omniglot/strokes_background'
args['data']['overfitting'] = False

train_dataset = OmniglotDataset(args['data'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
print("[main]: Iterations per epoch: ", len(train_loader))

loss_func = VAELowerBoundLoss()
# loss_func = evidence_lower_bound
model = VAE()
model = model.to('cuda')

# device = 'cuda'
# encoder = GaussianMLP(
#         input_dim=28*28, hidden_dim=500, output_dim=512
#     ).to(device)
# decoder = BernoulliDecoder(
#             latent_dim=512,
#             hidden_dim=500,
#             output_dim=28*28,
#         ).to(device)
# model = VAESample(
#         device=device,
#         encoder=encoder,
#         decoder=decoder,
#     ).to(device)

def save_image(x, y):
    image_x = x.detach().cpu().numpy().transpose((0,2,3,1))
    image_y = y.detach().cpu().numpy().transpose((0,2,3,1))
    image_x = (image_x * 255).astype(np.uint8)
    image_y = (image_y * 255).astype(np.uint8)
    for i in range(y.shape[0]):
        cv2.imshow('image_y', image_y[i])
        cv2.imshow('image_x', image_x[i])
        k = cv2.waitKey()
        
optimizer = Adam(model.parameters(), lr=0.0004, weight_decay=1e-6)
EPOCH = 100

for epoch in range(EPOCH):
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
        
        # x_hat, mean, logvar = model(net_in)
        # x_hat = x_hat.view(B, C, H, W)
        # likelihood_params = {'theta': x_hat}
        # input = {
        #         'log_var': logvar,
        #         'mu': mean,
        #         'x': sample['image'],
        #         'likelihood_params': likelihood_params
        #         }
        
        loss = loss_func(input)
        
        print(loss.item())
        
        loss.backward()
        optimizer.step()
    
save_image(sample['image'], y) 
# save_image(sample['image'], x_hat) 

