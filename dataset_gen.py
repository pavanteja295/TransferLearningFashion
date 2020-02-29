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
    delta_h = max(dh - img.size[1], 0)
    delta_w = max(dw - img.size[0], 0 )
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    new_im = ImageOps.expand(img, padding)
    return new_im


class Fashion_Dataset(Dataset):
    def __init__(self, dataset, split, dir_, transforms=None, finetune=False, debug=False, resize=(224, 224)):
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
        self.class_inst_list = data_[ft_pt]['class_inst']
        self.resize = tuple(resize[:2])

        
        if debug:
            #
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            
        else:
            if split == 'train':
                crop_ = [torchvision.transforms.CenterCrop, torchvision.transforms.RandomCrop, torchvision.transforms.RandomResizedCrop]
                crop_select = crop_[resize[2]](self.resize)
                
 
                self.transforms = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4), 
                                                                torchvision.transforms.RandomHorizontalFlip(), 
                                                                crop_select,
                                                                torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                                ]) # 
            else:
                self.transforms = torchvision.transforms.Compose([
                                                                torchvision.transforms.CenterCrop(self.resize),
                                                                torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        print('Number of  {} samples for {}  is {}'.format(split, 'finetune' if finetune else 'pretrain', len(self.data['image_list'])))
    
    def __getitem__(self, idx):

        img_p = os.path.join(self.dir_, self.data['image_list'][idx])
        img = Image.open(img_p).convert('RGB')
        img = pad_(self.resize[0], self.resize[1],  img)  #totensor already normalizes
        gt_ = self.data['class'][idx]
        
        # apply augmentations
        img_t = self.transforms( img)
        # pil_t = torchvision.transforms.ToPILImage()
        # img_tmp = pil_t(img_t)
        # img_tmp.save(str(idx) , 'JPEG')
        return img_t, gt_

    def __len__(self):
        return len(self.data['image_list'])


        





