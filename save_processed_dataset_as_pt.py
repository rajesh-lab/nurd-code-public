# Copyright (c) aahlad puli
# typical call:
# python save_processed_dataset_as_pt.py --img_side=3 --dataset=synthetic --nr_batch_size=200 --seed=9000 --workers=0 --hosp --label_balance_method=downsample --rho=0.9 --rho_test=0.9 --debug=10

from tensorboardX import SummaryWriter
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from tqdm import tqdm
import utils
from utils import concat

import numpy as np
import PIL
import torch
import torch.utils.data
from argparse import ArgumentParser

import sys
from os.path import dirname
filepath = os.path.abspath(__file__)
erm_dir = dirname(filepath)
parentdir = dirname(erm_dir)
sys.path.append(parentdir)
from dataloaders import construct_dataloaders, XYZ_DatasetWithIndices, CAN_LOAD_WATERBIRDS

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# device = "cuda"


def cli_main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    utils.add_args(parser, 'config')
    utils.add_args(parser, 'logging')
    utils.add_args(parser, 'dataset')
    parser.add_argument('--nr_batch_size', type=int, default=128, help='mini batch size')
    parser.add_argument('--store_name', type=str, default=None)
    parser.add_argument('--max_count', type=int, default=-1)
    args = parser.parse_args()
    # if args.eval_only:
    # assert args.store_name is not None

    assert args.train_filename is None, "This is for generation; cant use train_filename."

    if args.store_name is None:
        args.store_name = utils.process_save_name(args)
    print("WILL SAVE WITH PREFIX ", args.store_name)

    if args.equalized_groups:
        assert args.dataset == "waterbirds", "This is only done for waterbirds"
    if args.dataset == "waterbirds":
        assert CAN_LOAD_WATERBIRDS
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hparams = utils.get_DATA_HPARAMS(args)
    hparams['pin_memory'] = False
    hparams['shuffle'] = False
    hparams['create_dataset'] = True

    print('STARTED GETTING DATA')
    loaders_dict = construct_dataloaders( args, hparams, z_loaders=False, load=False)

    import time
    count = 0
    total = len(loaders_dict['train'].dataset)

    max_counts = {
        'train': args.max_count,
        'val': int(0.3*args.max_count),
        'test': int(0.3*args.max_count),
        'eval': int(0.3*args.max_count),
    }

    assert "train" in loaders_dict.keys()

    data_savename = "SAVED_DATA/{}.pt".format(args.store_name)
    save_dict={}

    for name, loader in loaders_dict.items():
        if name == "split":
            continue
        # data_savename = args.store_name + "_{}.pt".format(name)
        count = 0
        total = len(loader.dataset)
        x_list = []
        y_list = []
        h_list = []
        time_now = time.time()
        for batch in loader:

            x = batch[0]
            y = batch[1]

            if args.hosp:
                h = batch[2]

            print("STARTING WITH {}/{} ----- in {:.3f} s".format(count,
                                                                 total, time.time() - time_now))
            time_now = time.time()

            x_list.append(x)
            y_list.append(y.view(-1, 1))
            if args.hosp:
                h_list.append(h.reshape(-1, 1))

            count += x.shape[0]
            print("DONE WITH {}/{} ----- in {:.3f} s".format(count,
                                                             total, time.time() - time_now))

            if args.max_count > 1 and count > max_counts[name]:
                print("EXITING AT {}/{}".format(count, max_counts[name]))
                break
            
        dataset_to_save = XYZ_DatasetWithIndices(
                                concat(x_list),
                                concat(y_list),
                                concat(h_list) if len(h_list) > 0 else None
                            )
        save_dict[name] = dataset_to_save
        del x_list, y_list, h_list

        print("DONE WITH {}".format(name))

        if name == "train":
            # split_save_name = args.store_name + "_{}_split.pt".format(name)
            LEN = len(dataset_to_save)
            perm = torch.randperm(LEN)
            m_train = int(0.8*LEN)
            train_indices = perm[:m_train]
            val_indices = perm[m_train:]
            split_dict = {
                'train' : train_indices,
                'val' : val_indices
            }
            save_dict["split"] = split_dict

    torch.save(save_dict, data_savename)
    print("DONE SAVING")

if __name__ == '__main__':
    cli_main()
