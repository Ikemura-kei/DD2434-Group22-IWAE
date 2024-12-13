import torch
from torch.utils.data import Dataset
from easydict import EasyDict
import os
import cv2
import numpy as np
from torchvision.transforms import ToTensor, Compose

class OmniglotDataset(Dataset):
    def __init__(self, args: EasyDict):
        super().__init__()
        self.args = args
        
        self.root_dir = args.root_dir
        
        if not isinstance(args.image_subdir, list):
            self.image_subdir = [os.path.join(self.root_dir, args.image_subdir)]
            self.stroke_subdir = [os.path.join(self.root_dir, args.stroke_subdir)]
        else:
            self.image_subdir = []
            self.stroke_subdir = []
            for i in range(len(args.image_subdir)):
                self.image_subdir.append(os.path.join(self.root_dir, args.image_subdir[i]))
                self.stroke_subdir.append(os.path.join(self.root_dir, args.stroke_subdir[i]))
                
        self.overfitting = args.overfitting
        
        assert self._check_data_exists(), 'Data does not exist! Stopping.'
        
        self.image_paths = self._load_paths()
        
        self.transform = Compose([ToTensor()])
        
        print("[OmniglotDataset]: Loaded {} samples".format(len(self.image_paths)))
        
    def __getitem__(self, index):
        if self.overfitting:
            index = index % 2
        image_path = self.image_paths[index]
        stroke_path = image_path.replace(self.args.image_subdir, self.args.stroke_subdir)
        
        # -- read image to numpy array --
        # -- TODO: check value range --
        image = cv2.imread(image_path, -1) # (H, W)
        image = cv2.resize(image, (28, 28), cv2.INTER_CUBIC)
        # print('raw image data range {}, {}, {}, {}'.format(image.min(), image.max(), image.shape, image.dtype))
        # random_numbers = np.random.rand(28, 28) * 255
        # print(random_numbers[0])
        # image = np.where(image > random_numbers, 255, 0)
        image = np.where(image > 1e-1, 255, 0).astype(np.uint8)
        
        # -- transformations on image (at least converting to torch tensor) --
        image = self.transform(image)
        # print('transformed image data range {}, {}, {}'.format(image.min(), image.max(), image.shape))
        
        # -- read stroke --
        stroke = np.zeros(200)
        
        # -- transformations on stroke --
        
        return {'image': image, 'stroke': stroke}
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_paths(self):
        image_paths = []
        
        for i in range(len(self.image_subdir)):
            image_subdir = self.image_subdir[i]
            
            for category_dir in os.listdir(image_subdir):
                category_dir = os.path.join(image_subdir, category_dir)
                
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
        for i in range(len(self.image_subdir)):
            image_subdir = self.image_subdir[i]
            stroke_subdir = self.stroke_subdir[i]
            images_exist = os.path.exists(image_subdir) 
            strokes_exist = os.path.exists(stroke_subdir)
            if not (images_exist and strokes_exist):
                return False
        return True
        