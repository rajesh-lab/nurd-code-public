# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from tqdm import tqdm

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from torch.nn import functional as F


from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append("/scratch/apm470/nuisance-orthogonal-prediction/code/nrd-xray")
from dataloaders import construct_dataloaders

sys.path.append("/scratch/apm470/nuisance-orthogonal-prediction/code/nrd-xray/COVIDomaly/")
from utils.NeuralNet import Generator

from tensorboardX import SummaryWriter

import pytorch_lightning as pl
from models import LitClassifier

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

device = "cuda"


def cli_main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--dry', action='store_true', help='run on subset')
    parser.add_argument('--img_side', type=int, default=224, help='run on images of size')
    parser.add_argument('--transform', type=str, default=None, help='Transform sample using this')
    parser.add_argument('--dataset', type=str, default="chexpert", help='Which dataset to do ERM')
    parser.add_argument('--prefix', type=str, default=None, help='folder organization')
    parser.add_argument('--transform_val', action='store_true', help='do transform on val also')
    parser.add_argument('--noise', action='store_true', help='add training noise to x')
    parser.add_argument('--z_dim', type=int, default=128, help='latent code for autoencoders')
    parser.add_argument('--batch_size', type=int, default=128, help='mini batch size')
    parser.add_argument('--print_interval', type=int, default=10, help='mini batch size')
    parser.add_argument('--seed', type=int, default=1234, help='seed for the run')
    parser.add_argument('--workers', type=int, default=1234, help='workers for the run')
    parser.add_argument('--model_type', type=str, default="small", help='use this model')
    parser.add_argument('--datasetTWO', type=str, default="mimic", help='use this dataset along with chexpert to construct joint')
    parser.add_argument('--train_filename', type=str, help='DUMMY;USELESS;NO', default=None)
    parser.add_argument('--rho', type=float, default=0.8, help='rho to create imbalance')
 
    
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    # parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()
        
    pl.seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
            
    hparams = {}
    hparams['lambda'] = 0.1
    hparams['d_steps_per_g_step'] = 1
    hparams['mlp_width'] = 512
    hparams['mlp_depth'] = 2
    hparams['pretrained'] = False
    hparams['subset'] = True
    hparams['upsample'] = False
    hparams['pin_memory'] = True
    hparams['weight_decay'] = 1e-6
    hparams['workers'] = args.workers

    hparams['input_shape'] = [1] + [args.img_side, args.img_side]
    hparams['batch_size'] = args.batch_size
    hparams['rho'] = args.rho
    if args.prefix is None:
        args.prefix = "__".join([args.dataset, "ae" if "COVID" in args.transform else args.transform , "side{}".format(args.img_side) , "transform_val", str(args.transform_val), "dummy", str(args.seed), "BS{}".format(args.batch_size), ""])
    
    print('STARTED GETTING DATA')
    train_loader, val_loaders = construct_dataloaders(args, hparams, z_loaders=False)
    
    print('STARTED LOADING TRANSFORM MODEL')

    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpus!=0) else "cpu")

    if args.transform is not None:
        if 'COVIDomaly' in args.transform:
            print(' ---- LOADING AUTOENCODER')
            print(' ---- LOADING AUTOENCODER')
            print(' ---- LOADING AUTOENCODER')
            print(' ---- LOADING AUTOENCODER')
            print(' ---- LOADING AUTOENCODER')
            tmodel = Generator(height=args.img_side, width=args.img_side, channel=1,  
                        device=device, ngpu=1,
                        ksize=5, z_dim=args.z_dim, learning_rate=0.0)
            tmodel.load_state_dict(torch.load(args.transform, map_location=device))
            transform_func=tmodel
        elif args.transform[:10] == "discretize":            
            print(' ---- LOADING DISCRETIZATION TRANSFORM')
            print(' ---- LOADING DISCRETIZATION TRANSFORM')
            print(' ---- LOADING DISCRETIZATION TRANSFORM')
            print(' ---- LOADING DISCRETIZATION TRANSFORM')
            print(' ---- LOADING DISCRETIZATION TRANSFORM')
            try:
                n_categories = int(args.transform[11:])
                print('N_CAT = ', n_categories)
            except:
                assert False, "{} is not only int; please enter strings like discretize_16".format(args.transform[11:])
            def transform_func(x):
                # [-1, 1] ---> [0, k]
                x_unit = 0.49*(0.5 + x) # [-1, 1] ---> [0, 1]
                return torch.floor(x_unit*n_categories)/n_categories # [0, 1) ---> [0, k-1] ---> [0, ])
        else:
            assert False, 'BRO, dont know how to handle gan'

        model = LitClassifier(args.learning_rate, transform_func=transform_func, transform_val=args.transform_val, noise=args.noise)
            
    else: 
        print(' ---- RUNNING PLAIN ERM') 
        print(' ---- RUNNING PLAIN ERM') 
        print(' ---- RUNNING PLAIN ERM') 
        print(' ---- RUNNING PLAIN ERM') 
        print(' ---- RUNNING PLAIN ERM') 
        model = LitClassifier(args.learning_rate, transform_func=None)
    # ------------
    # data
    # ------------
    # dm = MNISTDataModule.from_argparse_args(args)
    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='pl_logs_allrun_again/{}'.format(args.prefix),
    )

    # ------------
    # model
    # ------------
    

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, progress_bar_refresh_rate=0,
        # limit_train_batches=0.2, 
        # limit_val_batches=0.2, 
        # precision=16
        )
    trainer.fit(model, train_loader, val_dataloaders=val_loaders)

    # ------------
    # testing
    # ------------
    # result = trainer.test(model, test_dataloaders=test_loader, limit_test)
    # print(result)


if __name__ == '__main__':
    # cli_lightning_logo()
    cli_main()
