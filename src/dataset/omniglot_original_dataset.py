import torch
from torch.utils.data import Dataset
from easydict import EasyDict
import os
import cv2
import numpy as np
from torchvision.transforms import ToTensor, Compose
import scipy.io


class OmniglotOriginalDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ = True
        
        self.root_dir = args.root_dir
        
        omni_raw = scipy.io.loadmat(
        os.path.join(self.root_dir, 'omniglot_original', 'chardata.mat'))

        def reshape_data(data):
            return data.reshape((-1, 28, 28))
        self.data = {'train': reshape_data(omni_raw['data'].T.astype('float32')), \
            'test': (omni_raw['testdata'].T.astype('float32'))}
        
        print("Num training data {}, num testing data {}".format(self.data['train'].shape, self.data['test'].shape))
        
    def train(self):
        self.train_ = True
        
    def eval(self):
        self.train_ = False
        
    def __len__(self):
        if self.train_:
            return len(self.data['train'])
        else:
            return len(self.data['test'])
        
    def __getitem__(self, idx):
        if self.train_:
            key = 'train'
        else:
            key = 'test'
            
        image = self.data[key][idx][None,...]
        image = torch.Tensor(image)
        
        return {'image': image}