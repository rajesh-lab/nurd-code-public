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
from dataloaders import construct_dataloaders

from tensorboardX import SummaryWriter

import pytorch_lightning as pl

from special_transforms import construct_transform

from sklearn.model_selection import train_test_split

def print_dict(dct):
    # print(" KEY : bal/unbal tells you whether the quantity was computed on balanced/unbalanced data; ")
    for item, val in dct.items():  # dct.iteritems() in Python 2
        print("{:25s} ({:.4f})".format(item, val))


def get_color(batch_x, y):
    c = torch.max(batch_x + 1, dim=2)[0]
    c = c.max(dim=2)[0]
    c = c.argmax(dim=1)
    batch_c = c.view(-1, 1, 1, 1)
    return batch_c.view(-1)

def generative_resample_dataset_for_synth(train_loader, args, phase='train'):
    dataset = train_loader.dataset
    x = dataset.tensors[0]
    y = dataset.tensors[1].long()
    z = x[:,2].long()

    perm = torch.randperm(y.shape[0])
    y_perm = y[perm]
    
    perm_2 = torch.randperm(y.shape[0])
    z_perm = z[perm_2]

    original = {
        '00' : [x[((1-y)*(1-z))>0], (1-y)*(1-z)],
        '10' : [x[(y*(1-z))>0], y*(1-z)],
        '01' : [x[((1-y)*z)>0], (1-y)*z],
        '11' : [x[(y*z)>0],  y*z]
    }

    permuted = {
        '00' : (1-y_perm)*(1-z_perm),
        '10' : y_perm*(1 - z_perm),
        '01' : (1 - y_perm)*z_perm,
        '11' : y_perm*z_perm
    }


    x_perm = torch.zeros(x.shape).to(x.device)
    
    for key, val in permuted.items():
        x_g_yz = original[key][0]
        original_count = x_g_yz.shape[0]
        resample_count = val.sum()
        resample_indices = torch.distributions.categorical.Categorical(torch.ones((original_count, ))).sample((resample_count,))

        # print(resample_indices)
        x_perm[val > 0] = x_g_yz[resample_indices,:]
    
    # y_perm = z_perm
    # pred_train = -x_perm[:, 0] + x_perm[:, 1]
    # y_pred = (pred_train > 0.5).long()
    # train_acc = (y_perm == y_pred).float().mean()
    # print("resampled train acc = ", train_acc)
    # print("resampled train acc = ", train_acc)
    # print("resampled train acc = ", train_acc)
    # print("resampled train acc = ", train_acc)
    # assert False

    h_dummy = torch.ones(y_perm.shape).to(y_perm.device)
    w = torch.ones(y_perm.shape).to(y_perm.device)/y_perm.shape[0]

    # print(w, w.sum())
       
    return_dataset = TensorDataset(x_perm, y_perm, h_dummy, w)

    return return_dataset


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

def one_loop(loader, model, phase="train", optimizer=None):

    meter = training_functions.AvgMeter()

    # print("mu network weight", model.mu_network.weight)
    # print("sigma network weight", model.sigma_network.weight)
    # print("sigma network   bias", model.sigma_network.bias)

    for idx, batch in enumerate(loader):
        y = batch[1]
        x = batch[0]
        z = x[:,2] # z is from inside
        yz = torch.cat([y.view(-1,1), z.view(-1,1)], dim=1)
        # print(yz.device, x.device)
        mu_pred, sigma_pred = model.forward(yz)
        loss = negative_gaussian_log_likelihood(mu_pred, sigma_pred, x[:,:2])
        meter.update(loss.item(), x.shape[0])

        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    print("sigma pred first 10 ", sigma_pred[:10])
        
    return meter.avg()

def train_and_sample_from_generative_model(train_val_loaders, args):
        
    gen_model = GenerativeMDN().to(args.device)
    generative_model_optimizer = torch.optim.Adam(gen_model.parameters(), lr=1e-3)

    train_loader, val_loader = train_val_loaders

    GENSAVEDIR = "pl_logs_REPRESENTATION/genmodel_{}".format(args.prefix)
    generative_model_saver = training_functions.SingleModelSaver(save_dir=GENSAVEDIR, args=args)

    for epoch in range(100):
        train_loss = one_loop(train_loader, gen_model, phase="train", optimizer=generative_model_optimizer) 
        val_loss = one_loop(val_loader, gen_model, phase="val", optimizer=None) 

        print("AT {}, TRAIN LOSS {} VAL LOSS {}".format(epoch, train_loss, val_loss))

        generative_model_saver.maybe_save(epoch, gen_model, val_loss)

    best_model = generative_model_saver.load_best(map_location=args.device, model=gen_model)

    train_dataset = train_loader.dataset
    x = train_dataset.tensors[0]
    y = train_dataset.tensors[1]
    z = x[:,2]

    perm = torch.randperm(y.shape[0])
    z_perm = z[perm]
    x_train_gen = best_model.sample(y, z_perm)
    x_train_gen = torch.cat([x_train_gen, z_perm.view(-1,1)], dim=1)

    val_dataset = val_loader.dataset
    x_val = val_dataset.tensors[0]
    y_val = val_dataset.tensors[1]
    z_val = x_val[:,2]

    perm = torch.randperm(y_val.shape[0])
    z_val_perm = z_val[perm]
    x_val_gen = best_model.sample(y_val, z_val_perm)
    x_val_gen = torch.cat([x_val_gen, z_val_perm.view(-1,1)], dim=1)

    train_dataset_gen = TensorDataset(x_train_gen, y)
    val_dataset_gen = TensorDataset(x_val_gen, y_val)

    return train_dataset_gen, val_dataset_gen

def cli_main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--dry', action='store_true', help='run on subset')
    parser.add_argument('--img_side', type=int, default=32, help='run on images of size')
    parser.add_argument('--pred_model_type', type=str, default="small", help='pred model type')

    parser.add_argument('--dataset', type=str, default="chexpert", help='Which dataset to do ERM')
    parser.add_argument('--prefix', type=str, help='folder organization', required=True)
    parser.add_argument('--noise', action='store_true', help='add training noise to x')
    parser.add_argument('--z_dim', type=int, default=128, help='latent code for autoencoders')
    parser.add_argument('--batch_size', type=int, default=128, help='mini batch size')
    parser.add_argument('--print_interval', type=int, default=10, help='mini batch size')
    parser.add_argument('--seed', type=int, default=1234, help='seed for the run')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers to use in loaders')
    parser.add_argument('--train_filename', type=str, default=None, help='use this dataset instead')
    parser.add_argument('--hosp_predict', action='store_true', help='Predict hospital from images')
    parser.add_argument('--rho', type=float, default=0.8, help='value or correlation for the joint dataset')
    parser.add_argument('--datasetTWO', type=str, default="mimic", help='use this dataset along with chexpert to construct joint')
    parser.add_argument('--debug',  type=int, default=0, help='print things; at 1 print only pred model stuff, at 2 print everything')
    parser.add_argument('--device', type=str, default="", help='DONT CHANGE')

    # representation arguments
    parser.add_argument('--minimax_batch_size', type=int, default=256, help='batchsize for represntation learning minimax')
    parser.add_argument('--lambda_', type=float, default=0.5, help='representation regularization loss')
    parser.add_argument('--max_lambda_', type=float, default=100, help='representation regularization max')

    default_lr=1e-3
    parser.add_argument('--theta_lr', type=float, default=default_lr, help='learning rate for theta in p_theta(Y | r_gamma(X) ) ')
    parser.add_argument('--gamma_lr', type=float, default=default_lr, help='learning rate for the representation')
    parser.add_argument('--phi_lr', type=float, default=default_lr, help='learning rate for phi in  p_phi(Y | r_gamma(X), Z ) ')
    parser.add_argument('--anneal', type=float, default=1.0, help='anneal addition')

    # parser.add_argument('--num_phi_steps', type=int, default=2, help='number of p_phi steps for represntation learning minimax')
    parser.add_argument('--frac_phi_steps', type=float, default=0.5, help='fraction on a epoch that p_phi steps')
    parser.add_argument('--randomrestart', action='store_true', help="restarts training of the phi model before every gamma theta step")
    parser.add_argument('--conditional', action='store_true', help="impose conditional independence only, otherwise this does joint independence")


    parser.add_argument('--augment', action='store_true', help="augment every batch half the time")
    parser.add_argument('--change_disease', type=str, help='disease to use instead of pneumonia', default=None) # only for joint. not used other wise
    parser.add_argument('--upsample_nr', action='store_true', help="upsample thing")
    parser.add_argument('--downsample_nr', action='store_true', help="downsample thing")
    parser.add_argument('--disc_groups', type=int, default=2, help="disc groups")
    parser.add_argument('--equalized_groups', action='store_true', help="equal props")

    parser.add_argument('--naug', type=str, default=None, help="type of naug", choices=["HG", "PR"])
    parser.add_argument('--patch_size', type=int, default=None, help='patch size in randomization')
    parser.add_argument('--sigma', type=float, default=None, help="add noise to the image")

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if args.dataset == 'cmnist':
        args.img_side = 28
            
    pl.seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.use_deterministic_algorithms(True)
    # if args.debug  > 0:
    #     torch.backends.cudnn.deterministic=True
    #     torch.backends.cudnn.benchmark=False

    args.device = torch.device("cuda" if (torch.cuda.is_available() and args.gpus!=0) else "cpu")

    DATA_HPARAMS = {}
    DATA_HPARAMS['subset'] = True
    DATA_HPARAMS['upsample'] = False
    DATA_HPARAMS['pin_memory'] = False
    DATA_HPARAMS['weight_decay'] = 1e-6
    DATA_HPARAMS['workers'] = args.workers
    DATA_HPARAMS['input_shape'] = [1] + [args.img_side, args.img_side]
    DATA_HPARAMS['batch_size'] = args.batch_size
    DATA_HPARAMS['rho'] = args.rho
    DATA_HPARAMS['hosp'] = True

    print('STARTED GETTING DATA')

    assert args.dataset not in ['chexpert', 'mimic'], "why would you reweight chexpert/mimic?"

    train_loader, val_loaders, z_train_loader, z_val_loader = construct_dataloaders(args, DATA_HPARAMS, z_loaders=True)
    test_loader = val_loaders[0] # test distribution loader
    val_loader = val_loaders[1] # train distribution heldout loader

    print("-------------------------------")


    train_dataset, val_dataset = train_and_sample_from_generative_model(train_val_loaders=[train_loader, val_loader], args=args)
    # train_x, val_x, train_y, val_y = train_test_split(samples_x.numpy(), samples_y.numpy(), test_size=0.2, train_size=0.8, shuffle=True)
    # train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_y), torch.ones(train_x.shape[0])/train_x.shape[0])
    # val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y), torch.from_numpy(val_y), torch.ones(val_x.shape[0])/val_x.shape[0])

    # assert False, val_dataset.tensors[3].sum()
    # val_dataset = generative_resample_dataset_for_synth(val_loader, args) # without args.platt, not val_iso created
    # train_dataset = generative_resample_dataset_for_synth(train_loader, args)

    WEIGHT_LOADER_BATCHSIZE=args.minimax_batch_size
    
    train_loader_without_weights = DataLoader(
        train_dataset, batch_size=WEIGHT_LOADER_BATCHSIZE, num_workers=DATA_HPARAMS['workers'], pin_memory=DATA_HPARAMS['pin_memory'], drop_last=False, shuffle=True)

    val_loader_without_weights = DataLoader(
        val_dataset, batch_size=WEIGHT_LOADER_BATCHSIZE, num_workers=DATA_HPARAMS['workers'], pin_memory=DATA_HPARAMS['pin_memory'], drop_last=False, shuffle=False)

    # val_loader_with_reversed_weights = DataLoader(
    #     reversed_val_dataset, batch_size=WEIGHT_LOADER_BATCHSIZE, num_workers=DATA_HPARAMS['workers'], pin_memory=DATA_HPARAMS['pin_memory'], drop_last=False, shuffle=False)

    # unbal_val_loader_with_uniform_weights = DataLoader(
        # unbalanced_dataset, batch_size=WEIGHT_LOADER_BATCHSIZE, num_workers=DATA_HPARAMS['workers'], pin_memory=DATA_HPARAMS['pin_memory'], drop_last=False, shuffle=False)

    def add_uniform_weights(loader):
        # function to add a weights vector the dataloader; adds 1/|data size| weight to every sample;
        # this ensures that when I sum weighted results across the dataset; I get the correct dataset average of a metric
        _dataset = loader.dataset
        x = _dataset.tensors[0]
        y = _dataset.tensors[1].view(-1)
        if args.dataset == "joint":
            h = _dataset.tensors[2].view(-1)
        else:
            h = y.view(-1)
        w = torch.ones(y.shape).view(-1)/y.shape[0]
        uniformly_weighted_data =  TensorDataset(x,y,h.to(y.device),w.to(y.device))
        return DataLoader(uniformly_weighted_data, batch_size=WEIGHT_LOADER_BATCHSIZE, num_workers=DATA_HPARAMS['workers'], pin_memory=DATA_HPARAMS['pin_memory'], drop_last=False, shuffle=False)

    _, unbalanced_val_and_test_loaders, _, _ = construct_dataloaders(args, DATA_HPARAMS, z_loaders=True)
    unbalanced_val_loader = unbalanced_val_and_test_loaders[1]

    test_loader_with_uniform_weights = add_uniform_weights(test_loader)
    unbal_val_loader_with_uniform_weights = add_uniform_weights(unbalanced_val_loader)

    train_loader_with_weights = add_uniform_weights(train_loader_without_weights)
    val_loader_with_weights = add_uniform_weights(val_loader_without_weights)

    # del weight_model_list

    # ------------
    # Y | X model
    # ------------
    
    pred_model_type = "linear" if args.dataset == "synthetic" else args.pred_model_type
    pred_model_type = "cmnist" if args.dataset=="cmnist" else pred_model_type

    if args.dataset == "cmnist":
        pred_model_transform_type = "identity" 
    elif args.dataset == "synthetic" :
        pred_model_transform_type = "synthetic_first_two"
    else:
        pred_model_transform_type = "onlycenter"

    # ============== MODEL DEFINITION =========================

    if args.dataset == "cmnist":
        # assert False, "usecase not done yet"
        pred_model_x_transform_type = "identity"
        pred_model_z_transform_type = "cmnist_color"
    elif args.dataset == "synthetic" :
        pred_model_x_transform_type = "synthetic_first_two"
        pred_model_z_transform_type = "synthetic_last"
    else:
        pred_model_x_transform_type = "onlycenter"
        pred_model_z_transform_type = "holemask"

    pred_model_hparams = dict(
        model_type=pred_model_type,
        reweighted=True,
        weight_model=False,
        verbose=args.debug,
        x_transform_type="identity" if args.dataset=="joint" else pred_model_x_transform_type,
        z_transform_type=pred_model_z_transform_type,
        model_stage='pred_model',
        out_dim=1,
        mid_dim=16,
        in_size=3  # this is fixed due to the synthetic data generation process
    )

    pred_model_hparams.update(vars(args))
    pred_model_bunch = training_functions.Bunch(pred_model_hparams)

    if args.conditional:
        assert False
        _RUN_LOOP = training_functions.one_representation_learning_loop
        _REG_LOSS_NAME = "kl_loss"
    else:
        _RUN_LOOP = training_functions.one_joint_independence_loop
        _REG_LOSS_NAME = "info_loss"
    
    SAVEDIR = "pl_logs_REPRESENTATION/repmodel_{}".format(pred_model_bunch.prefix)
    models = training_functions.create_models(pred_model_bunch)
    writer = SummaryWriter(logdir=SAVEDIR)
    
    # optimizer setup
    optimizers = training_functions.configure_optimizers(models, pred_model_bunch)

    saver = training_functions.ModelSaver(save_dir=SAVEDIR, args=pred_model_bunch)
    pred_model_bunch.current_lambda = pred_model_bunch.lambda_
    # the anneal ensures that max_lambda is reached in 0.5 epochs
    print("REPLACING ANNEAL")
    pred_model_bunch.anneal = 2*(pred_model_bunch.max_lambda_ - pred_model_bunch.lambda_) / pred_model_bunch.max_epochs

    # training and validating
    for epoch in range(args.max_epochs):
        # outerloop

        # inner loop
        train_metrics = _RUN_LOOP("train", train_loader_with_weights, models, pred_model_bunch, optimizers)
        # update current_lambda
        pred_model_bunch.current_lambda = pred_model_bunch.current_lambda + pred_model_bunch.anneal
        # ensure lambda is not more than max_lambda_
        pred_model_bunch.current_lambda = max(0, min(pred_model_bunch.current_lambda, pred_model_bunch.max_lambda_))

        if pred_model_bunch.debug > 0 :
            print(" ONE LOOP END with LAMBDA = {:.3f}".format(pred_model_bunch.current_lambda ))
            print(train_metrics)
        
        writer.add_scalars("train", train_metrics, global_step=epoch)
        
        val_metrics = _RUN_LOOP("val", val_loader_with_weights, models, pred_model_bunch, None)
        writer.add_scalars("val", val_metrics, global_step=epoch)

        unbal_val_metrics = _RUN_LOOP("val", unbal_val_loader_with_uniform_weights, models, pred_model_bunch, None)
        writer.add_scalars("unbal_val", unbal_val_metrics, global_step=epoch)

        # reversed_val_metric = _RUN_LOOP("val", val_loader_with_reversed_weights, models, pred_model_bunch, None)
        # writer.add_scalars("rev_val", reversed_val_metric, global_step=epoch)
        
        val_loss = args.max_lambda_*val_metrics[_REG_LOSS_NAME] + val_metrics['pred_loss']

        assert val_metrics[_REG_LOSS_NAME]>= 0, val_metrics[_REG_LOSS_NAME]
        saver.maybe_save(epoch, models, metric_val=val_loss)

        print("DONE WITH EPOCH {} ".format(epoch))
        print_dict(val_metrics)

        print("DEBUG : INTERMEDIATE TEST RESULTS")
        test_results = _RUN_LOOP("val", test_loader_with_uniform_weights, models, pred_model_bunch, None)
        print_dict(test_results)

        # print("DEBUG : INTERMEDIATE REV VAL RESULTS")
        # reversed_val_metric = _RUN_LOOP("val", val_loader_with_reversed_weights, models, pred_model_bunch, None)
        # print_dict(reversed_val_metric)


        print("=====================================================")

    writer.close()

    # Fix representations and train pred and aux models
    print("=====================================================")
    print("========= TRAINING ONLY REGULARIZER NOW =============")
    print("========= TRAINING ONLY REGULARIZER NOW =============")
    print("========= TRAINING ONLY REGULARIZER NOW =============")
    print("========= TRAINING ONLY REGULARIZER NOW =============")
    print("========= TRAINING ONLY REGULARIZER NOW =============")
    print("=====================================================")
    
    models = saver.load_best(map_location=pred_model_bunch.device)
    models["x_representation"].eval()
    models["pred_layer"].eval()

    print("AFTER LOADING BEST, VAL AND TEST RESULTS")
    print("AFTER LOADING BEST, VAL AND TEST RESULTS")
    print("AFTER LOADING BEST, VAL AND TEST RESULTS")
    print("AFTER LOADING BEST, VAL AND TEST RESULTS")
        
    print("DEBUG : INTERMEDIATE VAL RESULTS")
    val_metrics = _RUN_LOOP("val", val_loader_with_weights, models, pred_model_bunch, None)
    print_dict(val_metrics)

    print("DEBUG : INTERMEDIATE TEST RESULTS")
    test_results = _RUN_LOOP("val", test_loader_with_uniform_weights, models, pred_model_bunch, None)
    print_dict(test_results)
    
    PREDSAVEDIR = "pl_logs_REPRESENTATION/predmodel_{}".format(pred_model_bunch.prefix)
    pred_writer = SummaryWriter(logdir=PREDSAVEDIR)
    
    # optimizer setup
    optimizers = training_functions.configure_optimizers(models, pred_model_bunch)
    # return [opt_theta, opt_gamma, opt_phi]
    pred_optimizers = [optimizers[2]]
    del optimizers[1]
    del optimizers[0]
    
    pred_saver = training_functions.ModelSaver(save_dir=PREDSAVEDIR, args=pred_model_bunch)
    
    print("ACTUAL PRED MODEL TRAINING START")
    # training and validating
    for epoch in range(20):
        # outerloop

        # NO inner loop
        train_metrics = _RUN_LOOP("train", train_loader_with_weights, models, pred_model_bunch, pred_optimizers, regularizer_only=True)
        pred_writer.add_scalars("train", train_metrics, global_step=epoch)

        val_metrics = _RUN_LOOP("val", val_loader_with_weights, models, pred_model_bunch, None)
        pred_writer.add_scalars("val", val_metrics, global_step=epoch)

        print("================================")
        print("DEBUG : INTERMEDIATE TRAINING RESULTS")
        print_dict(train_metrics)
        print("DEBUG : INTERMEDIATE VALIDATION RESULTS")
        print_dict(val_metrics)
        print("================================")

        if args.conditional:
            pred_val_loss = val_metrics['aux_loss']
        else:
            pred_val_loss = val_metrics['critic_loss']

        pred_saver.maybe_save(epoch, models, metric_val=pred_val_loss)

    pred_writer.close()

    models = pred_saver.load_best(map_location=pred_model_bunch.device)
    for _, model in models.items():
        model.eval()

    # ------------
    # testing
    # ------------
    
    result_dicts = {}

    with torch.no_grad():
        train_results = _RUN_LOOP("val", train_loader_with_weights, models, pred_model_bunch, None)
        result_dicts["train"] = train_results

        val_results = _RUN_LOOP("val", val_loader_with_weights, models, pred_model_bunch, None)
        result_dicts["bal_val"] = val_results

        unbal_val_results = _RUN_LOOP("val", unbal_val_loader_with_uniform_weights, models, pred_model_bunch, None)
        result_dicts["unbal_val"] = unbal_val_results

        # reversed_val_metric = _RUN_LOOP("val", val_loader_with_reversed_weights, models, pred_model_bunch, None)
        # result_dicts["rev_val"] = reversed_val_metric

        test_results = _RUN_LOOP("val", test_loader_with_uniform_weights, models, pred_model_bunch, None)
        result_dicts["test"] = test_results

    result_dict = {}
    
    for key_name, dict_item in result_dicts.items():
        for key, val in dict_item.items():
            if "debug" in key:
                continue
            result_dict[ "{}___{}".format(key_name, key)] = val

    results_pt_filename = "pl_logs_REPRESENTATION/results_{}.pt".format(args.prefix)
    print(" SAVING RESULTS IN {}".format(results_pt_filename))
    print(" TEST RESULTS :")
    print(" KEY : train/test tell you which distribution the heldout set correponds to; ")
    print_dict(result_dict)

    torch.save(
        {
            # 'weight_model' : [],
            'pred_models' : result_dict
        },
        results_pt_filename
    )

if __name__ == '__main__':
    cli_main()
