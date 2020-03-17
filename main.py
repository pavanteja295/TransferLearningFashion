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

import torchvision
from torchvision import transforms
from bidict import bidict


def main():
    parser = argparse.ArgumentParser(description='Fashion Dataset')
    parser.add_argument('--dataset', dest='dataset', default='fashion-dataset', type=str,
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
    parser.add_argument('--epochs',  type=int, default=120,
                        help="epochs one for pretrain one for finetune")

    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--finetune_lr', type=float, default=0.001, help="Learning rate")
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
    parser.add_argument('--add_sampler', type=int, default=0,
                        help="Sampling for imbalance data")
    parser.add_argument('--num_workers', dest='num_workers', default=1, type=int,
                        help="Num data workers")
    parser.add_argument('--debug',  default=False, action='store_true', 
                        help='debug')
    parser.add_argument('--save_after',  default=30, type=int,
                        help='epochs to save model after')
    parser.add_argument('--dir_', dest='dir_', default='/h,', type=str,
                        help="Exp name to be added to the suffix")
    parser.add_argument('--freeze', dest='freeze',  default=False, action='store_true',
                        help="Freeze the pretrain model")    # give resnets layer
    parser.add_argument('--finetune_freeze', dest='finetune_freeze', default=False, action='store_true',
                        help="Freeze the finetune model") 
    parser.add_argument('--gamma', dest='gamma', default=0.2, type=float,
                        help="Lr drop")    # give resnets layer
    parser.add_argument('--resize', nargs="+", type=int, default=[224, 224, 2],
                        help="resize images")    # give resnets layer
                        
    parser.add_argument('--train_between', dest='train_between', default=False, action='store_true',
                        help="train between") 
    parser.add_argument('--freeze_layers', nargs="+", type=str, default=['fc.'],
                        help="which layers to freeze")
    parser.add_argument('--switch_all', type=int, default=0,
                        help="when to switch to all")
    parser.add_argument('--test',  dest='test', default='pretrain_test', type=str,
                        help="when to switch to all")

    args = parser.parse_args()
    debug = args.debug

    print(args.freeze_layers)
    datasets_ = {}

    # create pretrain gna fine tune datsets
    datasets_['train_pt'] = Fashion_Dataset(args.dataset, 'train', dir_=args.dir_, debug=debug, resize=args.resize)
    datasets_['test_pt'] = Fashion_Dataset(args.dataset, 'test', dir_=args.dir_, debug=debug, resize=args.resize)
    datasets_['train_ft'] = Fashion_Dataset(args.dataset, 'train', dir_=args.dir_, finetune=True, debug=debug, resize=args.resize)
    datasets_['test_ft'] = Fashion_Dataset(args.dataset, 'test', dir_=args.dir_, finetune=True, debug=debug, resize=args.resize)


    # This dict stores the dataloaders of finetune and pre-train
    dataloaders_ = {}
    for key, val in datasets_.items():
        if 'test' in key:
            # Sampler is turned off for the testing phase
            dataloaders_[key] = DataLoader(datasets_[key], batch_size=args.batch_size, 
                                      num_workers=args.num_workers)
        else:
            sampler = None
            if args.add_sampler:
                # Weighted sampler, instance are sampled based on total_count[ class i ] / total_num of examples
                dict_samp = datasets_[key].class_count
                wts = [ val for keu, val in dict_samp.items()]
                wts = [1 / wt if wt else 0 for wt in wts ]
                wts = torch.FloatTensor(wts)

                # weights for the samples
                sample_wts = [wts[t] for t in datasets_[key].data['class']]
                sampler = torch.utils.data.WeightedRandomSampler(sample_wts, len(sample_wts))
                dataloaders_[key] = DataLoader(datasets_[key], batch_size=args.batch_size, 
                                    sampler=sampler, num_workers=args.num_workers)
            else:
                # Suffle only no sampler
                dataloaders_[key] = DataLoader(datasets_[key], batch_size=args.batch_size, 
                                    sampler=sampler, num_workers=args.num_workers, shuffle=True)

    # all parameters passed to the network
    config = {'loaders' : {'pretrain' : [dataloaders_['train_pt'], dataloaders_['test_pt']] , 
                           'finetune' : [dataloaders_['train_ft'], dataloaders_['test_ft']]},
               'gpuid': args.gpuid, 'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
               'optimizer':args.optimizer,  'exp_name' : args.exp_name, 'nesterov':args.nesterov, 'model_name': args.model_name, 'pretrain_in':args.pretrain_in,
               'model_weights':args.model_weights, 'loss': args.loss, 'save_after':args.save_after, 'freeze': args.freeze, 'gamma' :args.gamma, 'debug':args.debug, 'finetune_lr':args.finetune_lr,
               'finetune_freeze':args.finetune_freeze, 'train_between' : args.train_between, 'freeze_layers': args.freeze_layers, 'switch_all':args.switch_all }      
     
    if debug:
        for key, val in dataloaders_.items():
            check_data(val, args.batch_size, key)
    
    net = Network_(config)
    
    if not args.model_weights and not args.only_finetune:
    # pre train 
        net.train_(args.epochs)
    else:
        print('------------------SKIPPING THE PRETRAN---------------')
    
    if args.test == 'finetune_test':
        net.str_ = 'finetune'
#        import pdb; pdb.set_trace()
        net.switch_finetune()
        net.load_model()
        acc, acc_5, acc_cl_1, acc_cl_5, losses  = net.validation(net.test_loader, 0)
    elif args.test == 'pretrain_test':
        net.str_ = 'pretrain'
        net.load_model()
        acc, acc_5, acc_cl_1, acc_cl_5, losses  = net.validation(net.test_loader, 0)

    else:
        net.load_model()
        net.train_(args.epochs, finetune=True)

    dict_json = {'acc_1': acc.avg, 'acc_5': acc_5.avg, 'acc_cl_1' : acc_cl_1, 'acc_cl_5':acc_cl_5 }
    print(dict_json)
    import json
    str_p = args.model_weights[:args.model_weights.rfind('/')]
    str_name = str_p[str_p.rfind('/')+1:]
    with open(f'{str_name}_result.json', 'w') as fp:
        json.dump(dict_json, fp)
    

if __name__ == '__main__':
    main()

