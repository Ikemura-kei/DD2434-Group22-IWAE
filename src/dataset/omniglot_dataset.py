import torch
from torch.utils.data import dataset
from easydict import EasyDict
import os
import cv2
import numpy as np

class OmniglotDataset(dataset):
    def __init__(self, args: EasyDict):
        self.args = args
        
        self.root_dir = args.root_dir
        self.image_subdir = os.path.joint(self.root_dir, args.image_subdir)
        self.stroke_subdir = os.path.joint(self.root_dir, args.stroke_subdir)
        
        assert self._check_data_exists(), 'Data does not exist! Stopping.'
        
        self.image_paths = self._load_paths()
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        stroke_path = image_path.replace(self.args.image_subdir, self.args.stroke_subdir)
        
        # -- read image to numpy array --
        # -- TODO: check value range --
        image = cv2.imread(image_path, -1) # (H, W)
        
        # -- transformations on image (at least converting to torch tensor) --
        
        # -- read stroke --
        stroke = np.zeros(200)
        
        # -- transformations on stroke --
        
        return {'image': image, 'stroke': stroke}
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_paths(self):
        image_paths = []
        
        for category_dir in os.listdir(self.image_subdir):
            category_dir = os.path.join(self.image_subdir, category_dir)
            
            if not os.path.isdir(category_dir):
                continue
            
            for character_dir in os.listdir(category_dir):
                character_dir = os.path.join(category_dir, character_dir)
                
                if not os.path.isdir(character_dir) or 'character' not in character_dir.split('/')[-1]:
                    continue
                
                for image_path in os.listdir(character_dir):
                    if not '.png' in image_path:
                        continue
                    
                    image_path = os.path.join(character_dir, image_path)
                    image_paths.append(image_path)
                    
        return image_paths
    
    def _check_data_exists(self):
        images_exist = os.path.exists(self.image_subdir) 
        strokes_exist = os.path.exists(self.stroke_subdir)
        return images_exist and strokes_exist
        