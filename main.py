"""
Author : Pavan Teja Varigonda
"""

import os
import argparse
import pandas as pd
import time
import glob
import json
from dataset_gen import Fashion_Dataset
import torchvision.models as models
from torch.utils.data import DataLoader
from default import Network_
import torch
#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True) for CIFAR 10
def main():
    parser = argparse.ArgumentParser(description='Fashion Dataset')
    parser.add_argument('--dataset', dest='dataset', default='fashion-dataset-small', type=str,
                        help=" select among fashion-dataset / fashion-dataset_small")
    parser.add_argument('--loss', dest='loss', default='cse', type=str,
                        help="Focal / Cross Entropy")
    parser.add_argument('--only_finetune', default=False, action='store_true',
                        help=" if not set trains a model from scratch and finetune")
    # parser.add_argument('--lr', default=0.01, action=float,
    #                     help="learning rate to use")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='resnet18',
                        help="resnet18/resnet50")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="Load model for finetune or test")
    parser.add_argument('--pretrain_in', default=False, action='store_true',
                        help="Load Imagenet weights")
    # for now just single value
    parser.add_argument('--epochs', nargs="+", type=int, default=120,
                        help="epochs one for pretrain one for finetune")

    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[60, 80, 120],)
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--workers', type=int, default=1, help="#Thread for dataloader")
    parser.add_argument('--nesterov',  default=False, action='store_true', help='nesterov up training phase')
    parser.add_argument('--exp_name', dest='exp_name', default='default', type=str,
                        help="Exp name to be added to the suffix")
    parser.add_argument('--add_sampler', default=False, action='store_true',
                        help="Sampling for imbalance data")
    parser.add_argument('--num_workers', dest='num_workers', default=1, type=int,
                        help="Num data workers")
    parser.add_argument('--debug',  default=False, action='store_true', 
                        help='debug')
    parser.add_argument('--save_after',  default=30, type=int,
                        help='epochs to save model after')
    parser.add_argument('--dir_', dest='dir_', default='/h,', type=str,
                        help="Exp name to be added to the suffix")
    args = parser.parse_args()
    debug = args.debug

    datasets_ = {}
    datasets_['train_pt'] = Fashion_Dataset(args.dataset, 'train', dir_=args.dir_, debug=debug)
    datasets_['test_pt'] = Fashion_Dataset(args.dataset, 'test', dir_=args.dir_, debug=debug)
    datasets_['train_ft'] = Fashion_Dataset(args.dataset, 'train', dir_=args.dir_, finetune=True, debug=debug)
    datasets_['test_ft'] = Fashion_Dataset(args.dataset, 'test', dir_=args.dir_, finetune=True, debug=debug)
    


    dataloaders_ = {}
    for key, val in datasets_.items():
        if 'test' in key:
            dataloaders_[key] = DataLoader(datasets_[key], batch_size=args.batch_size, 
                                      num_workers=args.num_workers)
        else:
            sampler = None
            if args.add_sampler:
                wts = [ val for keu, val in datasets_[key].class_count.items()]
                wts = 1 / torch.Tensor(wts)
                wts = wts.double()

                sampler = torch.utils.data.WeightedRandomSampler(wts, args.batch_size)
                dataloaders_[key] = DataLoader(datasets_[key], batch_size=args.batch_size, 
                                    sampler=sampler, num_workers=args.num_workers)
            else:
                dataloaders_[key] = DataLoader(datasets_[key], batch_size=args.batch_size, 
                                    sampler=sampler, num_workers=args.num_workers, shuffle=True)


    config = {'loaders' : {'pretrain' : [dataloaders_['train_pt'], dataloaders_['test_pt']] , 
                           'finetune' : [dataloaders_['train_ft'], dataloaders_['test_ft']]},
               'gpuid': args.gpuid, 'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
               'optimizer':args.optimizer,  'exp_name' : args.exp_name, 'nesterov':args.nesterov, 'model_name': args.model_name, 'pretrain_in':args.pretrain_in,
               'model_weights':args.model_weights, 'loss': args.loss, 'save_after':args.save_after }      
    
    if debug:
        for key, val in dataloaders_.items():
            check_data(val, args.batch_size, key)
    

    net = Network_(config)
    if not args.model_weights:
    # pre train 
        net.train_(args.epochs)
    # fine tune
    net.train_(args.epochs, finetune=True)

def check_data(data_loader, num, str_):
    import matplotlib.pyplot as plt
    import numpy as np
    dataiter = iter(data_loader)
    dir_ = os.path.join('viz', str_)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    for ind in range(10):
        
        images, labels = dataiter.next()
        images = images.numpy()

        labels = labels.numpy()

    # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(num):
            ax = fig.add_subplot(1,num, idx+1, xticks=[], yticks=[])
            plt.imshow(np.transpose(images[idx], (1, 2, 0)))
            ax.set_title(data_loader.dataset.class_list.inverse[labels[idx]])
        fig.savefig(os.path.join(dir_, str(ind) +'.png'  ))
        plt.close()




if __name__ == '__main__':
    main()

