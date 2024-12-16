from easydict import EasyDict
import cv2
import os
import numpy as np

def dict2edict(input_dict: dict):
    edict = EasyDict()
    
    for k, v in input_dict.items():
        if not isinstance(v, dict):
            edict[k] = v
        else:
            edict[k] = dict2edict(v)
            
    return edict

def save_images(images, save_dir, suffix=''):
        
    for b in range(len(images)):
        image = (images[b].detach().cpu().numpy().squeeze() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, '{}_{}.png'.format(b, suffix)), image)