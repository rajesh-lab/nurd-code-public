# Copyright (c) aahlad puli
# 


from aggregate_results import loss_func
import copy
import argparse
import collections
import json
import os
import random
from re import A
import sys
import time
import uuid
import csv
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, random_split
from argparse import ArgumentParser
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from utils import *

from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LogisticRegression as LR

from torch.nn import functional as F
import os.path
import matplotlib
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import TensorBoardLogger
import training_functions
import models

import sys
# sys.path.append("/scratch/apm470/generative-inpainting-pytorch/ERM/DomainBed")
sys.path.append("/scratch/apm470/nuisance-orthogonal-prediction/code/nrd-xray")
# from datasets import CheXpertDataset, MimicCXRDataset
from dataloaders import XYZ_DatasetWithIndices, construct_dataloaders

from tensorboardX import SummaryWriter

import pytorch_lightning as pl

from special_transforms import construct_transform

from sklearn.model_selection import train_test_split

import utils

from dataloaders import create_loader


_SAVEFOLDER = "LOGS"
_SAVEFOLDER_MODELS = "SAVED_MODELS"
EPS=1e-4


# def generative_resample_dataset_for_synth(train_loader, args, phase='train'):
#     dataset = train_loader.dataset
#     x = dataset.tensors[0]
#     y = dataset.tensors[1].long()
#     z = x[:,2].long()

#     perm = torch.randperm(y.shape[0])
#     y_perm = y[perm]
    
#     perm_2 = torch.randperm(y.shape[0])
#     z_perm = z[perm_2]

#     original = {
#         '00' : [x[((1-y)*(1-z))>0], (1-y)*(1-z)],
#         '10' : [x[(y*(1-z))>0], y*(1-z)],
#         '01' : [x[((1-y)*z)>0], (1-y)*z],
#         '11' : [x[(y*z)>0],  y*z]
#     }

#     permuted = {
#         '00' : (1-y_perm)*(1-z_perm),
#         '10' : y_perm*(1 - z_perm),
#         '01' : (1 - y_perm)*z_perm,
#         '11' : y_perm*z_perm
#     }


#     x_perm = torch.zeros(x.shape).to(x.device)
    
#     for key, val in permuted.items():
#         x_g_yz = original[key][0]
#         original_count = x_g_yz.shape[0]
#         resample_count = val.sum()
#         resample_indices = torch.distributions.categorical.Categorical(torch.ones((original_count, ))).sample((resample_count,))

#         # print(resample_indices)
#         x_perm[val > 0] = x_g_yz[resample_indices,:]
    
#     # y_perm = z_perm
#     # pred_train = -x_perm[:, 0] + x_perm[:, 1]
#     # y_pred = (pred_train > 0.5).long()
#     # train_acc = (y_perm == y_pred).float().mean()
#     # print("resampled train acc = ", train_acc)
#     # print("resampled train acc = ", train_acc)
#     # print("resampled train acc = ", train_acc)
#     # print("resampled train acc = ", train_acc)
#     # assert False

#     h_dummy = torch.ones(y_perm.shape).to(y_perm.device)
#     w = torch.ones(y_perm.shape).to(y_perm.device)/y_perm.shape[0]

#     # print(w, w.sum())
       
#     return_dataset = XYZ_DatasetWithIndices(x_perm, y_perm, z, w)

#     return return_dataset


class GenerativeMDN(nn.Module):
    def __init__(self):
        super().__init__()

        self.mu_network = models.TwoLayerMLP(in_size=2, out_size=2, mid_size=16)
        self.sigma_network = models.TwoLayerMLP(in_size=2, out_size=2, mid_size=16)

    def forward(self, yz):
        mu_pred = self.mu_network(yz)
        sigma_pred = F.softplus(self.sigma_network(yz))
        return mu_pred, sigma_pred
    
    def sample(self, y,z):
        yz = torch.cat([y.view(-1,1), z.view(-1,1)], dim=1)
        mu = self.mu_network(yz)
        sigma = F.softplus(self.sigma_network(yz))
        sigma = torch.diag_embed(sigma, offset=0, dim1=-2, dim2=-1)

        sampler = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)
        return sampler.sample()

def negative_gaussian_log_likelihood(mu_pred, sigma_pred, x):
    # THIS x is only the first two coordinates of what x is in the dataloader
    assert x.shape == mu_pred.shape
    mean_diff_sqrd = torch.pow(mu_pred - x,2)
    neg_log_prob = (mean_diff_sqrd/sigma_pred + torch.log(sigma_pred)).sum(dim=1)
    # distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu_pred, sigma_pred)
    # print(distribution.log_prob(x).shape)
    return neg_log_prob.sum()

def one_loop(loader, model, device, phase="train", optimizer=None):
    meter = training_functions.AvgMeter()

    # print("mu network weight", model.mu_network.weight)
    # print("sigma network weight", model.sigma_network.weight)
    # print("sigma network   bias", model.sigma_network.bias)
    for batch in loader:
        y = batch[1].to(device)
        x = batch[0].to(device)
        z = x[:,2].to(device) # z is from inside
        yz = torch.cat([y.view(-1,1), z.view(-1,1)], dim=1).to(device)
        # print(yz.device, x.device)
        mu_pred, sigma_pred = model.forward(yz)
        # print(mu_pred.device, x.device, sigma_pred.device)
        loss = negative_gaussian_log_likelihood(mu_pred, sigma_pred, x[:,:2])
        meter.update(loss.item(), x.shape[0])

        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    # print("sigma pred first 10 ", sigma_pred[:10])
        
    return meter.avg()

def train_and_sample_from_generative_model(train_val_loaders, args, dataset, split_dict):
        
    gen_model = GenerativeMDN().to(args.device)
    generative_model_optimizer = torch.optim.Adam(gen_model.parameters(), lr=1e-3)

    train_loader = train_val_loaders[0]
    val_loader = train_val_loaders[1]

    GENSAVEDIR = "LOGS/genmodel_{}".format(args.prefix)
    generative_model_saver = training_functions.SingleModelSaver(save_dir=GENSAVEDIR, args=args)

    gen_model.to(args.device)
    for epoch in range(args.nr_epochs):
        gen_model.train()
        train_loss = one_loop(train_loader, gen_model, device=args.device, phase="train", optimizer=generative_model_optimizer) 

        gen_model.eval()
        val_loss = one_loop(val_loader, gen_model, device=args.device, phase="val", optimizer=None) 

        print("AT {}, TRAIN LOSS {} VAL LOSS {}".format(epoch, train_loss, val_loss))

        generative_model_saver.maybe_save(epoch, gen_model, val_loss)

    gen_model.cpu()

    best_model = generative_model_saver.load_best(map_location=args.device, model=gen_model)

    train_split = split_dict['train']
    val_split = split_dict['val']
    all_x = dataset.x
    all_y = dataset.y
    all_z = dataset.z
    
    # [train_split]
    # train_dataset = train_loader.dataset
    x = all_x[train_split]
    y = all_y[train_split].view(-1)
    z = x[:,2]

    perm = torch.randperm(y.shape[0])
    z_perm = z[perm]
    x_train_gen = best_model.sample(y, z_perm)
    x_train_gen = torch.cat([x_train_gen, z_perm.view(-1,1)], dim=1)

    x_val = all_x[val_split]
    y_val = all_y[val_split].view(-1)
    z_val = x_val[:,2]

    perm = torch.randperm(y_val.shape[0])
    z_val_perm = z_val[perm]
    x_val_gen = best_model.sample(y_val, z_val_perm)
    x_val_gen = torch.cat([x_val_gen, z_val_perm.view(-1,1)], dim=1)

    train_dataset_gen = XYZ_DatasetWithIndices(x_train_gen, y, z_perm)
    val_dataset_gen = XYZ_DatasetWithIndices(x_val_gen, y_val, z_val_perm)

    return train_dataset_gen, val_dataset_gen

def cli_main():

    # ------------
    # args
    # ------------
        # ------------
    # args
    # ------------
    parser = ArgumentParser()
    utils.add_args(parser, 'config')
    utils.add_args(parser, 'logging')
    utils.add_args(parser, 'dataset')
    utils.add_args(parser, 'model')
    utils.add_args(parser, 'training')
    utils.add_args(parser, 'optimization')
    utils.add_args(parser, 'nuisance')
    parser.add_argument('--load_generated_data', action='store_true', help='load saved generated data.', default=False)
    args = parser.parse_args()
    assert args.prefix is not None, "args.prefix is required."
    assert args.rho_test_for_eval is None, "this will overwrite the test set being loaded; please use the evaluation script if you want to just evaluate. you can also set rho_test."
    
    # ------------
    # fix seeds
    # ------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.use_deterministic_algorithms(True)


    # ------------
    # some sanity checks
    # ------------
    utils.SANITY_CHECKS_ARGS(args)
    # configuring variables for running this script
    dataset_to_side_map = {"cmnist": 28, "joint": args.img_side,
                           "chexpert": 32, "mimic": 32, "waterbirds": args.img_side}

    if args.dataset in dataset_to_side_map.keys():
        args.img_side = dataset_to_side_map[args.dataset]

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------
    # DATALOADERS
    # ------------
    DATA_HPARAMS = utils.get_DATA_HPARAMS(args)
    print('STARTED GETTING DATA')

    assert args.dataset not in [
        'chexpert', 'mimic'], "why would you reweight chexpert/mimic?"

    datasets_dict = construct_dataloaders(args, DATA_HPARAMS, z_loaders=True, load=True, return_datasets_only=True)

    _split = datasets_dict['split']
    train_split = _split['train']
    val_split = _split['val']

    LOADER_HPARAMS = {   
        'BATCH_SIZE' : args.nr_batch_size,
        'workers' : args.workers,
        'pin_memory' : True
    }

    train_loader = create_loader(Subset(datasets_dict['train'], train_split), LOADER_HPARAMS)
    val_loader = create_loader(Subset(datasets_dict['train'], val_split), LOADER_HPARAMS)

    print("-------------------------------")


    generated_save_pt_filename = "{}/generated_{}.pt".format(_SAVEFOLDER, args.prefix)
    if os.path.isfile(generated_save_pt_filename) and args.load_generated_data:
        loaded_data = torch.load(generated_save_pt_filename)
        train_dataset = loaded_data['train']
        val_dataset = loaded_data['val']
    else:
        train_dataset, val_dataset = train_and_sample_from_generative_model(
                                                                    train_val_loaders=(train_loader, val_loader), 
                                                                    args=args,
                                                                    dataset=datasets_dict['train'],
                                                                    split_dict=_split)
        # train_x, val_x, train_y, val_y = train_test_split(samples_x.numpy(), samples_y.numpy(), test_size=0.2, train_size=0.8, shuffle=True)
        # train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_y), torch.ones(train_x.shape[0])/train_x.shape[0])
        # val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y), torch.from_numpy(val_y), torch.ones(val_x.shape[0])/val_x.shape[0])

        # assert False, val_dataset.tensors[3].sum()
        # val_dataset = generative_resample_dataset_for_synth(val_loader, args) # without args.platt, not val_iso created
        # train_dataset = generative_resample_dataset_for_synth(train_loader, args)
        # weights = weights.cpu()
        # BALANCE THE WEIGHTS HERE AND ALSO MATCH THE WEIGHT FOR EACH CLASS EQUAL ITS MARGINAL PROBABILITY
        # if not args.exact:
        #     # we only save for learned weights
        #     print(" SAVING WEIGHTS IN {}".format(weights_save_pt_filename))
        
        y_train_long = train_dataset.y.long().view(-1)
        y_val_long = val_dataset.y.long().view(-1)
        y_test_long = datasets_dict['test'].y.long().view(-1)

        train_weights = torch.ones_like(y_train_long).float().to(y_train_long.device)
        val_weights = torch.ones_like(y_val_long).float().to(y_val_long.device)

        train_dataset.weights = utils.multiple_by_labelmarginal_and_balance_weights(train_weights, y_train_long, y_weights_from_array=y_test_long)
        val_dataset.weights = utils.multiple_by_labelmarginal_and_balance_weights(val_weights, y_val_long, y_weights_from_array=y_test_long)

        torch.save(
            {
                'train': train_dataset,
                'test': val_dataset,
            },
            generated_save_pt_filename
        )

    if args.nr_only:
        if args.load_generated_data:
            return
        print("SKIPPING DISTILLATION; ADD --load_generated_data to use the saved data and run distillation.")
        return # skips the rest
    

    DIST_LOADER_PARAMS = {}
    DIST_LOADER_PARAMS['BATCH_SIZE'] = args.dist_batch_size
    DIST_LOADER_PARAMS['workers'] = args.workers
    DIST_LOADER_PARAMS['pin_memory'] = True

    weighted_train_loader = create_loader(
        train_dataset,
        DIST_LOADER_PARAMS,
        shuffle=True,
        weights=train_weights, # thitorhcs is used to produce a sampler because we cannot access weights from subset easily.
        strategy=args.nr_strategy
    )

    weighted_train_loader_for_evaluation_only = create_loader(
        val_dataset,
        DIST_LOADER_PARAMS,
        shuffle=False,
        strategy="weight"
    )

    weighted_val_loader = create_loader(
        val_dataset,
        DIST_LOADER_PARAMS,
        shuffle=False,
        strategy="weight"
    )
    
    print("====================================")
    print("========= DISTILLATION =============")
    print("========= DISTILLATION =============")
    print("========= DISTILLATION =============")
    print("====================================")

    assert args.pred_model_type is not None
    # if args.dataset == "synthetic", "'linear' or not"
    pred_model_type = args.pred_model_type

    # ============== DISTILLATION =========================
    # ============== DISTILLATION =========================
    # ============== DISTILLATION =========================

    pred_model_z_transform_type, pred_model_x_transform_type = utils._PRED_MODEL_NUISANCE_TYPES(args)

    pred_model_hparams = dict(
                                model_type=pred_model_type, reweighted=True, weight_model=False,
                                verbose=args.debug,
                                x_transform_type=pred_model_x_transform_type,
                                z_transform_type=pred_model_z_transform_type,
                                model_stage='pred_model',
                                out_dim=1, mid_dim=16, in_size=3  # this is fixed due to the synthetic data generation process
                            )

    pred_model_hparams.update(vars(args))
    pred_model_hparams['patch_size'] = args.patch_size
    pred_model_bunch = utils.Bunch(pred_model_hparams)

    # model, optimizer, and saver setup
    DISTSAVEDIR = "{}/repmodel_{}{}".format(_SAVEFOLDER_MODELS, pred_model_bunch.prefix, "" if args.add_pred_suffix is None else args.add_pred_suffix)
    pred_model_bunch.current_lambda = pred_model_bunch.lambda_

    # the anneal ensures that max_lambda is reached in 0.5 epochs
    # print("REPLACING ANNEAL")
    if args.max_lambda_ < 1e-3:
        pred_model_bunch.anneal = 0
    else:
        pred_model_bunch.anneal = 2 * \
            (pred_model_bunch.max_lambda_ - pred_model_bunch.lambda_) / \
            pred_model_bunch.dist_epochs

    # default create_loader is no weighting or sampling
    dist_loaders = {
        'train': weighted_train_loader,
        'tr_fix' : weighted_train_loader_for_evaluation_only,
        'val': weighted_val_loader,
        'unbal_val': train_loader,
        'test': create_loader(datasets_dict['test'], DIST_LOADER_PARAMS),
    }

    if args.eval:
        dist_loaders['eval'] = create_loader(datasets_dict['eval'], DIST_LOADER_PARAMS)

    dist_saver = training_functions.ModelSaver(save_dir=DISTSAVEDIR, args=pred_model_bunch)

    # assert False, "COMMENT THIS TO ENABLE INNER LOOP"
    FINALSAVEDIR = "{}/finalmodel_{}{}".format(
        _SAVEFOLDER_MODELS, pred_model_bunch.prefix, "" if args.add_pred_suffix is None else args.add_pred_suffix)
    final_saver = training_functions.ModelSaver(save_dir=FINALSAVEDIR, args=pred_model_bunch)

    # just a function to call to evaluate a model
    evaluate_model_on_dist_loaders = lambda _model: training_functions.run_one_epoch(
                                        training_functions.one_joint_independence_loop, 
                                        pred_model_bunch,
                                        dist_loaders,
                                        _model,
                                        optimizers=None,
                                        phase="val",
                                        objective='dist')

    if args.load_dist_model:
        # dist_model = final_saver.load_best(map_location=pred_model_bunch.device, objective='dist')
        dist_model = dist_saver.load_best(map_location=pred_model_bunch.device, objective='dist')
        print("LOADED PRE-CRITIC DISTILLATION MODEL.")
    elif args.load_final_model:
        dist_model = final_saver.load_best(map_location=pred_model_bunch.device, objective='dist')
        print("LOADED POST-CRITIC DISTILLATION MODEL.")
    else:
        dist_model, _ = training_functions.fit_model(
            args_bunch=pred_model_bunch,
            epochs=pred_model_bunch.dist_epochs,
            loaders=dist_loaders,
            objective='dist',
            writer=SummaryWriter(logdir=DISTSAVEDIR),
            saver=dist_saver
        )

        print("DISTILLATION COMPLETE.")
        
    training_functions.PUT_MODELS_ON_DEVICE(dist_model, args.device)

    dist_results = evaluate_model_on_dist_loaders(dist_model)
    utils.print_dict_of_dicts(dist_results)
    print("")

    if args.max_lambda_ < 1e-3 or args.dont_do_critic:
        final_results = dist_results
        final_saver = dist_saver
    else:
        # Fix representations and train pred and aux model
        print("=====================================================")
        print("========= TRAINING ONLY REGULARIZER NOW =============")
        print("========= TRAINING ONLY REGULARIZER NOW =============")
        print("========= TRAINING ONLY REGULARIZER NOW =============")
        print("=====================================================")

        # this will take loaded model and fit
        _, final_results = training_functions.fit_model(
            args_bunch=pred_model_bunch,
            epochs=pred_model_bunch.dist_epochs*2,
            loaders=dist_loaders,
            objective='critic_only',
            writer=SummaryWriter(logdir=FINALSAVEDIR),
            saver=final_saver,
            loaded_model=dist_model
        )

    del dist_model

    # ------------
    # conluding, printing, and saving
    # ------------

    if not args.dont_save_final_results:
        results_pt_filename = "{}/results_{}{}.pt".format(_SAVEFOLDER, args.prefix, "" if args.add_pred_suffix is None else args.add_pred_suffix)
        print(" SAVING RESULTS IN {}".format(results_pt_filename))
        utils.print_dict_of_dicts(final_results)

        torch.save(
            {
                'final_results': final_results,
                'final_modelpath': final_saver.best_path,
                'args' : args
            },
            results_pt_filename
        )
    else:
        print('NOT SAVING RESULTS')

if __name__ == '__main__':
    cli_main()
