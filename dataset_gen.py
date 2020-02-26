"""
Author : Pavan Teja Varigonda
"""

import os
import argparse
import pandas as pd
import time
import glob
import json
from torch.utils.data import DataLoader, Dataset
import torchvision
from PIL import Image, ImageOps
import cv2
from bidict import bidict


def pad_(dh, dw, img):
    delta_h = dh - img.size[1]
    delta_w = dw - img.size[0]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))  
    # print(delta_h, delta_w)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    new_im = ImageOps.expand(img, padding)
    return new_im


class Fashion_Dataset(Dataset):
    def __init__(self, dataset, split, dir_, transforms=None, finetune=False, debug=False):
        super(Fashion_Dataset, self).__init__()
        cwd = os.getcwd()
        dict_path = os.path.join(cwd,'data', dataset, 'data.json')
        with open(dict_path, 'r') as f:
            data_ = json.load(f)

        if finetune:
            ft_pt = 'finetune'
        else:
            ft_pt = 'pretrain'

        self.dir_ = dir_
        self.data = data_[ft_pt][split]
        self.class_list =  bidict(data_[ft_pt]['class_list'])
        self.class_count = data_[ft_pt]['total_class_count']
        
        if debug:
            #
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            
        else:
            if split == 'train':
                self.transforms = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(hue=.05, saturation=.05), 
                                                                torchvision.transforms.RandomHorizontalFlip(), 
                                                                torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            else:
                self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        print('Number of  {} samples for {}  is {}'.format(split, 'finetune' if finetune else 'pretrain', len(self.data['image_list'])))
    
    def __getitem__(self, idx):

        img_p = os.path.join(self.dir_, self.data['image_list'][idx])
        img = Image.open(img_p).convert('RGB')
        img = pad_(80, 60, img)  #totensor already normalizes
        gt_ = self.data['class'][idx]

        # apply augmentations
        img_t = self.transforms( img)
        return img_t, gt_

    def __len__(self):
        return len(self.data['image_list'])


        





