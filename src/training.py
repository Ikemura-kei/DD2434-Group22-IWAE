# Training 

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from datetime import datetime

#from IWAE import *
from iwae_model import *
from iwae_loss import *

from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
# EPOCHS = 3280
K = 1


args = EasyDict()

args['network'] = EasyDict()
args['network']['EncoderNetwork'] = EasyDict()
args['network']['EncoderNetwork']['input_dimensions'] = 28 * 28
args['network']['EncoderNetwork']['hidden_layers'] = [200, 200]
args['network']['EncoderNetwork']['latent_dimensions'] = 50
args['network']['EncoderNetwork']['activation'] = "tanh"
args['network']['EncoderNetwork']['k'] = 3
# K=1: 72.5224
# K=5: 68.9538
# K=50: 67.4221
args['network']['DecoderNetwork'] = EasyDict()
args['network']['DecoderNetwork']['output_dimension'] = 28 * 28
args['network']['DecoderNetwork']['hidden_layers'] = [200, 200]
args['network']['DecoderNetwork']['latent_dimension'] = 50
args['network']['DecoderNetwork']['activation'] = "tanh"
args['network']['DecoderNetwork']['output_activation'] ="sigmoid"

args['train'] = EasyDict()
args['train']['batch_size'] = 20
args['train']['optimizer_params'] = EasyDict()
args['train']['optimizer_params']['betas'] = (0.9, 0.999)
args['train']['optimizer_params']['eps'] = 1e-4
args['train']['optimizer_params']['lr'] = 1e-3

args['test'] = EasyDict()
args['test']['batch_size'] = 60
args['test']['eval_period'] = 98400
# args['test']['eval_period'] = 1000


save_dir = os.path.join('./logs', datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
os.makedirs(save_dir, exist_ok=True)
import json
with open(os.path.join(save_dir, 'cfg.json'), 'w') as f:
    json.dump(args, f, indent=4)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=args.train.batch_size, shuffle=True)
eval_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
eval_data_loader = DataLoader(eval_dataset, batch_size=args.test.batch_size, shuffle=False)

loss_func = IWAELowerBoundLoss()
model = IWAE(args['network'])
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), **args.train.optimizer_params)

print(DEVICE)

I = 7 ############ 7 for 3280 EPOCH
total_iterations = sum([3**i for i in range(I+1)]) * len(data_loader)
accum_iter = 0
print('Total iterations:',total_iterations)
print('Evaluation set size:', len(eval_data_loader))
print('Train set size:', len(data_loader))
       
with tqdm(total=total_iterations) as pbar:  
    for i in range(I+1):
        lr_scaler = 10 ** (-i/7.0)
        new_lr = args['train']['optimizer_params']['lr'] * lr_scaler

        # -- lr scheduling --
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        
        num_iterations = 3 ** i
        
        for sub_iteration in range(num_iterations):

            for it, sample in enumerate(data_loader):
                image = sample[0].to(DEVICE)  # The image tensor
                label = sample[1].to(DEVICE)  # The label tensor
                
                B, C, H, W = image.shape
                # flattened_img = image.view(image.size(0), 1, -1)
                flattened_img = image.view(-1, 1, 28 * 28)

                optimizer.zero_grad()
                mean, logvar, z, y = model(flattened_img)

                input = {
                            'log_var': logvar,
                            'mu': mean,
                            'x': flattened_img,
                            'recon_x': y,
                            'repar_z': z,
                            'k': K
                            }
                loss = loss_func(input)

                loss.backward()
                optimizer.step()
                        
                pbar.update(1)
                accum_iter += 1
                if it % 1000 == 0:
                    pbar.set_description("Loss {:.2f}, lr: {:.5f}, iter: {:.1f}".format(loss.item(), new_lr,accum_iter))
                    
                    
                if accum_iter % args['test']['eval_period'] == 0:
                    model.eval()
                    model.encoder.k = 5000
                    with torch.no_grad():
                        eval_loss = 0
                        n_samples = 0
                        for _, sample in enumerate(eval_data_loader):
                            image = sample[0].to(DEVICE)  # The image tensor
                            label = sample[1].to(DEVICE)  # The label tensor
                            
                            B, C, H, W = image.shape
                            flattened_img = image.view(-1, 1, 28 * 28)

                            mean, logvar, z, y = model(flattened_img)

                            input = {
                                        'log_var': logvar,
                                        'mu': mean,
                                        'x': flattened_img,
                                        'recon_x': y,
                                        'repar_z': z,
                                        'k': 5000
                                        }
                            loss = loss_func(input)
                            eval_loss += (loss * sample[0].shape[0])
                            n_samples += sample[0].shape[0]
                            # print(loss)

                        eval_loss = eval_loss / n_samples
                        print("eval loss", eval_loss)
                        torch.save(model.state_dict(), os.path.join(save_dir, "iwae_model_iter{:07d}_nll{:.3f}.pth".format(accum_iter, eval_loss.item())))
                    model.encoder.k = args['network']['EncoderNetwork']['k']
                    model.train()


torch.save(model.state_dict(), os.path.join(save_dir,"iwae_model.pth"))
print("Training complete.")

trained_iwae = IWAE(args['network'])  # Recreate the model object
trained_iwae.load_state_dict(torch.load('iwae_model.pth'))  
trained_iwae = trained_iwae.to(DEVICE)



def plot_reconstructions(iwae_model, input, SEED=1):
    np.random.seed(SEED)
    iwae_model.to(DEVICE)

    n = 5
    i_samples = np.random.choice(range(len(input)), n, replace=False)

    plt.figure(figsize=(10, 4))
    plt.suptitle("Reconstructions", fontsize=16, y=1, fontweight='bold')
    
    for counter, i_sample in enumerate(i_samples):
        
        # Plotting input image
        original, _ = input[i_sample]
        ax = plt.subplot(2, n, 1 + counter)
        plt.imshow(original[0].squeeze(0), vmin=0, vmax=1, cmap='gray')
        plt.axis('off')
        if counter == 0:
            ax.annotate("Input", xy=(-0.1, 0.5), xycoords="axes fraction",
                        va="center", ha="right", fontsize=12)

        # Plotting IWAE reconstruction
        flattened = original.view(-1)
        _, _, _, y = iwae_model(flattened.unsqueeze(0).to(DEVICE)) 
        y = y.unsqueeze(0).view(28,28).detach().cpu().numpy()
        ax = plt.subplot(2, n, 1 + counter + n)
        plt.imshow(y, vmin=0, vmax=1, cmap='gray')
        plt.axis('off')
        if counter == 0:
            ax.annotate("IWAE", xy=(-0.1, 0.5), xycoords="axes fraction",
                        va="center", ha="right", fontsize=12)

    return

plot_reconstructions(trained_iwae, dataset)


