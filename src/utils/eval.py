import torch

import os

def evaluation(model, test_loader, save_dir, eval_metric, accum_iter):
    model.eval()
                    
    original_k = model.encoder.k
    model.encoder.k = 5000
    with torch.no_grad():
        eval_loss = 0
        n_samples = 0
        for _, sample in enumerate(test_loader):
            image = sample[0]
            image = image.to('cuda')
            
            net_in = image.view(-1, 1, 28 * 28)

            mean, logvar, z, y = model(net_in)

            input = {
                    'log_var': logvar,
                    'mu': mean,
                    'x': net_in,
                    'recon_x': y,
                    'repar_z': z,
                    'k': 5000
                    }
            loss = eval_metric(input)
            eval_loss += (loss * image.shape[0])
            n_samples += image.shape[0]

        eval_loss = eval_loss / n_samples
        # print("eval loss", eval_loss)
        torch.save(model.state_dict(), os.path.join(save_dir, "model_iter{:07d}_nll{:.3f}.pth".format(accum_iter, eval_loss.item())))
        
    model.encoder.k = original_k
    return eval_loss.item(), image, y