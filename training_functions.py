# training_functions.py
import torchvision.transforms as transforms
from models import *
from utils import waterbirds_list
from re import I
from threading import current_thread
# from typing import NewType
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
import copy
import os
from special_transforms import construct_transform
import utils
import logging

import sys
from os.path import dirname
filepath = os.path.abspath(__file__)
erm_dir = dirname(filepath)
parentdir = dirname(erm_dir)
sys.path.append(parentdir)


train_transform = transforms.Compose([
    transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255]),
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.RandomResizedCrop(size=(224, 224)),
    transforms.RandomAffine(
        degrees=(0, 180), translate=(0.1, 0.2), scale=(0.9, 1.0)),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                         std=[0.229, 0.224, 0.255]),
])

xray_transform = transforms.Compose([
    #  transforms.RandomResizedCrop(size=(224, 224)),
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.RandomAffine(
        degrees=(-15, 15), translate=(0.0, 0.05), scale=(0.95, 1.0)),
])


EMPTY_METRICS_FOR_JOINT_INDEPENDENCE = {
    "pred_loss": 0,
    "critic_loss": 0,
    "info_loss": 0,
    "acc": 0,
    "critic_acc": 0,
    "critic_acc_debug": 0
}


class ModelSaver(object):

    def __init__(self, save_dir, args, max=False):
        super(ModelSaver, self).__init__()
        self.save_dir = save_dir
        self.ckpt_basename = ""
        self.best_metric_val = None
        self.best_epoch = 0
        self.args = args
        self.max = max
        self.type = "max" if self.max else "min"

        suffix = '_best.pth.tar'
        self.best_path = os.path.join(self.save_dir, suffix)
        print("best model path is", self.best_path)

    def _is_best(self, metric_val):
        if metric_val is None:
            return False

        if self.max:
            return self.best_metric_val is None or metric_val >= self.best_metric_val
        else:
            return self.best_metric_val is None or metric_val <= self.best_metric_val

    def maybe_save(self, epoch, model, metric_val, print_flag=True):
        if self._is_best(metric_val):
            self.best_metric_val = metric_val
            self.best_epoch = epoch
            ckpt_dict = {'args': self.args}
            ckpt_dict['epoch'] = epoch
            ckpt_dict['metric'] = metric_val
            for key, _model in model.items():
                model_copy = copy.deepcopy(_model)
                model_copy.to('cpu')
                ckpt_dict[key +
                          '_model_state'] = copy.deepcopy(model_copy.state_dict())
            torch.save(ckpt_dict, self.best_path)
            if print_flag:
                print("NEW {} BEST AT EPOCH {: <5} WITH LOSS {:.3f}".format(
                    self.type, epoch, metric_val))
                print("")
            return True
        else:
            if print_flag:
                print("EPOCH {} WITH LOSS {:.3f} NOT THE BEST COMPARED TO {:.3f}".format(
                    epoch, metric_val, self.best_metric_val))
            return False

    def load_best(self, map_location, objective="dist"):

        ckpt_dict = torch.load(self.best_path, map_location=map_location)
        best_epoch = ckpt_dict.get('epoch', -1)
        best_metric_val = ckpt_dict.get('metric', None)
        print("")
        print("")
        print("LOADING FROM {} WHICH CORRESPONDS TO EPOCH {} WITH LOSS {}".format(
            self.best_path, best_epoch, best_metric_val))
        print("")
        print("")
        hparams = ckpt_dict['args']
        # print(hparams)
        if objective == "dist":
            model = create_model(hparams)
        elif objective == "erm":
            model = create_erm_model(hparams)
        else:
            assert False, "WHAT IS {}?".format(objective)
        for key, _model in model.items():
            _model.load_state_dict(ckpt_dict[key + '_model_state'])
        return model


class SingleModelSaver(object):

    def __init__(self, save_dir, args):
        super(SingleModelSaver, self).__init__()
        self.save_dir = save_dir
        self.ckpt_basename = ""
        self.best_metric_val = None
        self.best_epoch = 0
        self.args = args

        suffix = '_best.pth.tar'
        # self.best_path = os.path.join(self.save_dir, suffix)
        self.best_path = self.save_dir + suffix
        print("best model path is", self.best_path)

    def _is_best(self, metric_val):
        if metric_val is None:
            return False
        return self.best_metric_val is None or metric_val < self.best_metric_val

    def maybe_save(self, epoch, model, metric_val):
        if self._is_best(metric_val):
            self.best_metric_val = metric_val
            self.best_epoch = epoch
            ckpt_dict = {'args': self.args}
            model_copy = copy.deepcopy(model)
            ckpt_dict['model_state'] = model_copy.to('cpu').state_dict()
            torch.save(ckpt_dict, self.best_path)
            print("NEW BEST AT EPOCH {} WITH LOSS {:.3f}".format(epoch, metric_val))
            print("SAVING AT {}".format(self.best_path))
        else:
            print("EPOCH {} WITH LOSS {:.3f} NOT THE BEST COMPARED TO {:.3f}".format(
                epoch, metric_val, self.best_metric_val))

    def load_best(self, map_location, model):
        print("LOADING FROM {} WHICH CORRESPONDS TO EPOCH {}".format(
            self.best_path, self.best_epoch))
        ckpt_dict = torch.load(self.best_path, map_location=map_location)
        model.load_state_dict(ckpt_dict['model_state'])
        return model

# METRIC ACCUMULATING CLASSES


class Meter:
    def __init__(self):
        self.N = 0
        self.total = 0

    def update(self, val):
        self.total += val

    def sum(self):
        return round(self.total, 4)


class MultiMeter(object):
    def __init__(self, metric_dict):
        self.meters = {key: Meter() for key in metric_dict.keys()}

    def update(self, metric_dict, N=None):
        # N is not used; just exists for common function calls
        for key, val in metric_dict.items():
            self.meters[key].update(val.item())

    def compile_result(self):
        return {key: meter.sum() for key, meter in self.meters.items()}


class AvgMeter:
    def __init__(self):
        self.N = 0
        self.total = 0
        self.N = 0

    def update(self, val, N):
        self.total += val
        self.N += N

    def avg(self):
        return round(self.total/self.N, 4)


class AvgMultiMeter(object):
    def __init__(self, metric_dict):
        self.meters = {key: AvgMeter() for key in metric_dict.keys()}

    def update(self, metric_dict, N):
        for key, val in metric_dict.items():
            self.meters[key].update(val.item(), N)

    def compile_result(self):
        return {key: meter.avg() for key, meter in self.meters.items()}


def process_batch(batch, hparams, device, return_weights, phase="val"):
    """ get x, y, and weight out of the batch

    Args:
        batch (_type_): batch
        hparams (_type_): parameters to determine what happens to the batch
        device (_type_): cuda or not
        phase (str, optional): tells you whether the batch will be used to train or not. Defaults to "val".
        return_weights (bool, optional): tells you whether the weights should be None, when we know the batch comes from a dataset without weights, this should be set to False. Defaults to True.

    Returns:
        _type_: _description_
    """
    # THIS FUNCTIONS ASSUMES THAT W IS DATASET NORMALIZED; SUM_{i \in dataset } W_i = 1
    x = batch[0].to(device)
    y = batch[1].view(-1).to(device)
    if hparams.hosp_predict:
        assert False
        y = batch[2].view(-1).to(device)
    # the third element of batch is the index string or the weight, this is why we use return_weights to ask for weights only when we use them
    w = batch[3].view(-1).to(device) if return_weights else None

    if phase == "train":
        # TRANSFORMS
        if hparams.augment:
            assert False
            if torch.randn(size=(1,)) < 0:
                if hparams.dataset in waterbirds_list:
                    x = train_transform(x)
                elif hparams.dataset == "joint":
                    x = xray_transform(x)
                else:
                    assert False, "DATA AUG FOR {}???".format(hparams.dataset)

        if hparams.noise and torch.randn(size=(1,)) < 0:
            # assert False
            x = x + 0.1*torch.randn(x.size(), device=x.device)
    
    return x, y, w


def process_erm_batch(batch, hparams, device, phase="val"):
    # WE ASSUME THAT W IS DATASET NORMALIZED; SUM_{i \in dataset } W_i = 1
    x = batch[0].to(device)
    y = batch[1].view(-1).to(device)
    if hparams.hosp_predict:
        assert False
        y = batch[2].view(-1).to(device)
    idx = batch[len(batch)-1].view(-1).to(device)

    # print((y.max(), y.min()))
    if phase == "train":
        # TRANSFORMS
        if hparams.augment and hparams.dataset in waterbirds_list:
            raise NotImplementedError("NOT USED IN EXPERIMENTS")
            if torch.randn(size=(1,)) < 0:
                x = train_transform(x)
        if hparams.noise and torch.randn(size=(1,)) < 0:
            # assert False
            x = x + 0.1*torch.randn(x.size(), device=x.device)

    return x, y, idx


def compute_loss(x, y, w, model):

    y_pred_g_rx, y_pred_g_rx_z = rep_forward_pass(model, x)

    # we will predict with  Y | r(X) only.
    y_pred = torch.max(y_pred_g_rx, dim=1)[1].view(-1)
    acc_vec = (y_pred == y).float().view(-1)

    y_g_rx_loss_vec = F.cross_entropy(
        y_pred_g_rx, y.long().view(-1), reduction='none').view(-1)
    y_g_rx_z_loss_vec = F.cross_entropy(
        y_pred_g_rx_z, y.long().view(-1), reduction='none').view(-1)

    # KL loss computation
    kl_loss_vec = -(y_g_rx_z_loss_vec - y_g_rx_loss_vec)

    metrics = {
        "pred_loss": torch.sum(y_g_rx_loss_vec*w.view(-1)),
        "aux_loss": torch.sum(y_g_rx_z_loss_vec*w.view(-1)),
        "kl_loss": torch.abs(torch.sum(kl_loss_vec*w.view(-1))),
        "acc": torch.sum(acc_vec*w.view(-1))
    }

    return metrics


def EVAL_MODELS(model):
    for _model in model.values():
        _model.eval()

def TRAIN_MODELS(model):
    for _model in model.values():
        _model.train()

def TRAIN_PRED_EVAL_CRITIC(model):
    
    model["x_representation"].train()
    model["pred_layer"].train()

    model["z_representation"].eval()
    model["critic_model"].eval()

def EVAL_PRED_TRAIN_CRITIC(model):

    model["x_representation"].eval()
    model["pred_layer"].eval()

    model["z_representation"].train()
    model["critic_model"].train()


def CPU_MODELS(model):
    for _model in model.values():
        _model.cpu()

def PUT_MODELS_ON_DEVICE(model, device):
    for _model in model.values():
        _model.to(device)

def create_erm_model(hparams):

    # this is mainly used to build a weight model for nuisances; set hparams.transform_type="identity" to build an actual ERM model
    assert hparams.weight_model
    model = {}
    # model setup
    if hparams.dataset == "synthetic":
        model["x_representation"] = FFRepresentation(
            hparams, _transform_type=hparams.transform_type, mid_dim=hparams.mid_dim).to(hparams.device)
    else:
        # r_gamma
        model["x_representation"] = ImageRepresentation(
            hparams, _transform_type=hparams.transform_type, mid_dim=None).to(hparams.device)

    # p_theta
    model['pred_layer'] = TopLayer(
        in_size=hparams.out_dim, out_size=2).to(hparams.device)
    return model


def configure_erm_optimizers(model, hparams):

    # theta, gamma, phi parameters
    theta = list(model['pred_layer'].parameters()) + \
        list(model['x_representation'].parameters())
    opt_theta = torch.optim.Adam(
        theta, lr=hparams.nr_lr, weight_decay=hparams.nr_decay)

    return [opt_theta]


def create_model(hparams):

    # we have two kinds of learning;
    # 1. conditional independence ; THIS IS DEPRECATED
    # p(Y | r(X) ) and model p_phi(Y | r(X), Z) =  aux model which has params for the representation s(Z) and phi.
    # 2. joint independence
    # p(Y | r(X) ) and model f_phi to classify between p(Y , r(X), Z) and p( Y , r(X)) p( Z ); f_phi has parameters for phi and s(Z)
    model = {}
    # model setup
    if hparams.dataset == "synthetic":
        model["x_representation"] = FFRepresentation(
            hparams, _transform_type=hparams.x_transform_type, mid_dim=hparams.mid_dim).to(hparams.device)
        model["z_representation"] = FFRepresentation(
            hparams, _transform_type=hparams.z_transform_type, mid_dim=hparams.mid_dim).to(hparams.device)
    else:
        # r_gamma
        model["x_representation"] = ImageRepresentation(
            hparams, _transform_type=hparams.x_transform_type, mid_dim=None, _for_critic_model=False).to(hparams.device)

        # representation of Z in p_phi
        model["z_representation"] = ImageRepresentation(
            hparams, _transform_type=hparams.z_transform_type, mid_dim=None, _for_critic_model=True).to(hparams.device)

    # if hparams.dataset == "synthetic":
        # with torch.no_grad():
        # print("DEBUG DEBUG DEBUG")
        # print("DEBUG DEBUG DEBUG")
        # print("DEBUG DEBUG DEBUG")
        # print("DEBUG DEBUG DEBUG")
        # print("DEBUG DEBUG DEBUG")
        # model["x_representation"].lin_rep.weight = nn.Parameter(torch.tensor([1,0,0]).float().view(1,-1).to(hparams.device))

    # p_theta
    model['pred_layer'] = TopLayer(
        in_size=hparams.out_dim, out_size=2).to(hparams.device)

    if hparams.conditional:
        raise NotImplementedError("THE CONDITIONAL CODE HAS NOT BEEN CHECKED")
        # last layer of p_phi
        model['aux_model'] = TopLayer(
            in_size=2*hparams.out_dim, out_size=2).to(hparams.device)
    else:
        # this is the critic model; takes in Y , r(X), s(Z)
        model['critic_model'] = TopLayer(
            in_size=2*hparams.out_dim + 1, out_size=2, nonlin=True).to(hparams.device)

    return model


def configure_optimizers(model, hparams):

    # theta, gamma, phi parameters
    theta = model['pred_layer'].parameters()
    gamma = model['x_representation'].parameters()

    if hparams.conditional:
        raise NotImplemented("FUNCTIONALITY NEEDS TO BE CLEANED AND DEBUGGED.")
        phi = list(model['aux_model'].parameters()) + \
            list(model['z_representation'].parameters())
    else:
        phi = list(model['critic_model'].parameters()) + \
            list(model['z_representation'].parameters())

    opt_theta = torch.optim.Adam(
        theta, lr=hparams.theta_lr, weight_decay=hparams.dist_decay)
    opt_gamma = torch.optim.Adam(
        gamma, lr=hparams.gamma_lr, weight_decay=hparams.dist_decay)

    # all parameters in p_phi( Y | r_gamma(X), Z ) other than gamma
    opt_phi = torch.optim.Adam(
        phi, lr=hparams.phi_lr, weight_decay=hparams.phi_decay)

    return [opt_theta, opt_gamma, opt_phi]


def rep_forward_pass(model, x):
    assert False
    x_rep = model["x_representation"].forward(x)
    y_pred_g_rx = model["pred_layer"].forward(x_rep)

    z_rep = model["z_representation"].forward(x)
    xz_rep = torch.cat([x_rep, z_rep], dim=1)
    y_pred_g_rx_z = model["aux_model"].forward(xz_rep)

    return y_pred_g_rx, y_pred_g_rx_z


def one_representation_learning_loop(phase, loader, model, hparams, optimizers, critic_only=False):
    raise NotImplementedError("THIS LOOP HAS NOT BEEN CHECKE TO WORK.")
    # model has 4 things
    # x_rep_model
    # z_rep_model
    # pred_layer
    # aux_model
    loader_copy = copy.deepcopy(loader)

    if phase != 'train':
        optimizers = None

    device = hparams.device

    # for k in range(args.K-1):
    metrics = {
        "pred_loss": 0,
        "aux_loss": 0,
        "kl_loss": 0,
        "acc": 0
    }

    meters = MultiMeter(metrics)
    assert False, "NO"
    for idx, batch in enumerate(loader):
        if phase == 'train':
            iter_loader = iter(loader_copy)
            total_batches = int(len(loader_copy)*hparams.frac_phi_steps)
            if critic_only:
                opt_phi = optimizers
                total_batches = 1
            else:
                opt_theta, opt_gamma, opt_phi = optimizers

            for batch_idx in range(total_batches):
                try:
                    batch = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(loader_copy)
                    batch = next(iter_loader)

                _x, _y, _w = process_batch(batch, hparams, device, phase=phase)
                phi_metrics = compute_loss(_x, _y, _w, model)
                opt_phi.zero_grad()
                phi_metrics["aux_loss"].backward()
                opt_phi.step()

                if hparams.debug > 2:
                    print(batch_idx, phi_metrics)

                # if batch_idx % 20 == 0:
                #     theta_metrics = compute_loss(_x, _y, _w, model=model)
                #     opt_theta.zero_grad()
                #     theta_metrics["pred_loss"].backward()
                #     opt_theta.step()

            # for joint dataset reset aux model weights
            if hparams.randomrestart and not critic_only:
                # when pred only the critic model only takes 1 step
                # reinitialize aux model and aux optimizers
                replace_model = create_model(hparams)
                replace_optimizers = configure_optimizers(
                    replace_model, hparams)
                model["critic_model"] = replace_model["critic_model"]
                model["z_representation"] = replace_model["z_representation"]
                optimizers[2] = replace_optimizers[2]
                opt_phi = replace_optimizers[2]

            x, y, w = process_batch(batch, hparams, device, phase=phase)
            if critic_only:
                assert False, "THIS NEEDS SOME THOUGHT"
                theta_metrics = compute_loss(x, y, w, model=model)
                opt_theta.zero_grad()
                loss = theta_metrics['aux_loss']
                loss.backward()
                opt_theta.step()

            else:
                theta_gamma_metrics = compute_loss(x, y, w, model=model)
                opt_gamma.zero_grad()
                opt_theta.zero_grad()

                loss = theta_gamma_metrics['pred_loss'] + \
                    hparams.current_lambda*theta_gamma_metrics['kl_loss']
                loss.backward()

                opt_gamma.step()
                opt_theta.step()

                # hparams.current_lambda = hparams.current_lambda + hparams.anneal
                # hparams.current_lambda = min(hparams.current_lambda, hparams.max_lambda_)

        with torch.no_grad():
            x, y, w = process_batch(batch, hparams, device, phase=phase)
            metrics = compute_loss(x, y, w, model)
            meters.update(metrics)

            if hparams.debug > 1:
                print("====================")
                print(" BATCH {} with LAMBDA = {:.3f}".format(
                    idx, hparams.current_lambda))
                print_dict(metrics)

                try:
                    print("=========================")
                    print("x_representation weights",
                          model['x_representation'].lin_rep.weight.data)
                except:
                    continue

    return meters.compile_result()

# JOINT INDEPDENDENCE CODE BELOW
# JOINT INDEPDENDENCE CODE BELOW
# JOINT INDEPDENDENCE CODE BELOW
# JOINT INDEPDENDENCE CODE BELOW
# JOINT INDEPDENDENCE CODE BELOW
# JOINT INDEPDENDENCE CODE BELOW
# JOINT INDEPDENDENCE CODE BELOW


def log_ratio_forward_pass(model, x, y, pred_only=False):
    x_rep = model["x_representation"].forward(x)
    y_pred_g_rx = model["pred_layer"].forward(x_rep)

    if pred_only:
        return y_pred_g_rx, None, None
    # this is sample from p( r(X), Y, Z)
    z_rep = model["z_representation"].forward(x)
    # z_rep = x[:, 2].view(-1,1)
    xyz_rep = torch.cat([x_rep, y.view(-1, 1), z_rep], dim=1)
    y_pred_g_rxyz = model["critic_model"].forward(xyz_rep)

    # this is sample from p( r(X), Y ) p(Z)
    perm = torch.randperm(z_rep.shape[0])
    xy_randz_repy = torch.cat([x_rep, y.view(-1, 1), z_rep[perm]], dim=1)
    y_pred_g_rx_rand_z = model["critic_model"].forward(xy_randz_repy)

    return y_pred_g_rx, y_pred_g_rxyz, y_pred_g_rx_rand_z


def log_ratio_forward_pass_ermonly(model, x, y):
    x_rep = model["x_representation"].forward(x)

    y_pred_g_rx = model["pred_layer"].forward(x_rep)

    return y_pred_g_rx


def acc_func(unnormed_vector_for_two_classes, target):
    # we will predict with  Y | r(X) only.
    y_pred = torch.max(unnormed_vector_for_two_classes, dim=1)[1].view(-1)
    return (y_pred == target).float().view(-1)


def compute_joint_independence_loss(x, y, w, model, pred_only=False, normalize=False):

    # this returns unweighted sum if w is None
    w_view1 = w.view(-1) if w is not None else torch.ones_like(y.float()).view(-1)
    y_pred_g_rx, y_pred_g_rxyz, y_pred_g_rx_rand_z = log_ratio_forward_pass(
        model, x, y, pred_only=pred_only)

    # y_pred = torch.max(y_pred_g_rx, dim=1)[1].view(-1) # we will predict with  Y | r(X) only.
    # acc_vec = (y_pred == y).float().view(-1)
    acc_vec = acc_func(y_pred_g_rx, y)
    y_g_rx_loss_vec = F.cross_entropy(
        y_pred_g_rx, y.long().view(-1), reduction='none').view(-1)

    # print(acc_vec.mean(), w_view1.sum(), torch.sum(acc_vec*w_view1))

    if pred_only:
        metrics = {
            "pred_loss": torch.sum(y_g_rx_loss_vec*w_view1),
            "acc": torch.sum(acc_vec*w_view1),
            "critic_loss": torch.sum(0*w_view1),
            "info_loss": torch.abs(torch.sum(0*w_view1)),
            "critic_acc": torch.sum(0*w_view1*0.5 + 0*w_view1*0.5),
            "critic_acc_debug": torch.sum(0*w_view1*0.5 + 0*w_view1*0.5)/torch.sum(w_view1)
        }
        if normalize:
            metrics = {
                "pred_loss": metrics["pred_loss"]/w_view1.sum(),
                "critic_loss": metrics["critic_loss"]/w_view1.sum(),
                "info_loss": metrics["info_loss"]/w_view1.sum(),
                "acc": metrics["acc"]/w_view1.sum(),
                "critic_acc": metrics["critic_acc"]/w_view1.sum(),
                "critic_acc_debug": metrics["critic_acc_debug"]/w_view1.sum(),
            }
        # FUNCTION EXITS HERE
        return metrics

    joint_labels = torch.ones(y.shape[0], device=y_pred_g_rxyz.device).long()
    prod_marg_labels = torch.zeros(
        y.shape[0], device=y_pred_g_rxyz.device).long()
    acc_vec_joint_vec = acc_func(y_pred_g_rxyz, joint_labels)
    acc_vec_prod_vec = acc_func(y_pred_g_rx_rand_z, prod_marg_labels)

    # loss for the pred model

    y_g_rx_z_loss_vec_joint = F.cross_entropy(
        y_pred_g_rxyz, joint_labels, reduction='none').view(-1)

    y_g_rx_z_loss_vec_marginal = F.cross_entropy(
        y_pred_g_rx_rand_z, prod_marg_labels, reduction='none').view(-1)

    critic_loss_vec = 0.5*(y_g_rx_z_loss_vec_joint +
                           y_g_rx_z_loss_vec_marginal)

    # compute log p(r(X), Y, Z is a joint sample) and log  p(r(X), Y, Z is NOT a joint sample), take difference and use the log-ratio trick to get
    p_y1_g_joint_samples_logprob_vec = torch.log_softmax(y_pred_g_rxyz, dim=1)[
        :, 1]
    p_y0_g_joint_samples_logprob_vec = torch.log_softmax(y_pred_g_rxyz, dim=1)[
        :, 0]
    information_loss_vec = p_y1_g_joint_samples_logprob_vec - \
        p_y0_g_joint_samples_logprob_vec

    metrics = {
        "pred_loss": torch.sum(y_g_rx_loss_vec*w_view1),
        "critic_loss": torch.sum(critic_loss_vec*w_view1),
        # "kl_loss": torch.abs(torch.sum(critic_loss_vec*w_view1)),
        "info_loss": torch.abs(torch.sum(information_loss_vec*w_view1)),
        "acc": torch.sum(acc_vec*w_view1),
        "critic_acc": torch.sum(acc_vec_joint_vec*w_view1*0.5 + acc_vec_prod_vec*w_view1*0.5),
        "critic_acc_debug": torch.sum(acc_vec_joint_vec*w_view1*0.5 + acc_vec_prod_vec*w_view1*0.5)/torch.sum(w_view1)
    }

    return metrics

def _pred_train_helper(optimizers, model, _batch, hparams, phase, device, pred_only):
    # is hparams.METERS != "weighted", w will be None
    x, y, w = process_batch(
    _batch, hparams, device, phase=phase, return_weights=hparams.METERS == "weighted")
    opt_theta, opt_gamma, opt_phi = optimizers

    # THE OUTER LOOP WITH REPRESENTATION AND PRED MODEL LEARNING. PHI PARAMETER UPDATES OF SIZE TOTAL_BATCHES
    opt_gamma.zero_grad()
    opt_theta.zero_grad()

    theta_gamma_metrics = compute_joint_independence_loss(
        x, y, w, model=model, pred_only=pred_only)

    if pred_only:
        loss = theta_gamma_metrics['pred_loss']
    else:
        loss = theta_gamma_metrics['pred_loss'] + \
            hparams.current_lambda*theta_gamma_metrics['info_loss']

    loss.backward()

    opt_gamma.step()
    opt_theta.step()

def _critic_train_helper(opt_phi, model, _batch, hparams, phase, device):
    # is hparams.METERS != "weighted", w will be None
    _x, _y, _w = process_batch(
        _batch, hparams, device, phase=phase, return_weights=hparams.METERS == "weighted")
    
    opt_phi.zero_grad()
    phi_metrics = compute_joint_independence_loss(
        _x, _y, _w, model)
    phi_metrics["critic_loss"].backward()

    opt_phi.step()

    return phi_metrics


def one_joint_independence_loop(phase, loader, model, hparams, optimizers=None, critic_only=False, warmup_critic_flag=False):
    # model has 4 things
    # x_rep_model
    # z_rep_model
    # pred_layer
    # critic_model
    if phase != 'train':
        optimizers = None
    
    pred_only = hparams.max_lambda_ < 1e-3
    loader_copy = copy.deepcopy(loader)

    device = hparams.device

    if hparams.METERS == "weighted":
        meters = MultiMeter(EMPTY_METRICS_FOR_JOINT_INDEPENDENCE)
    elif hparams.METERS == "plain_avg":
        meters = AvgMultiMeter(EMPTY_METRICS_FOR_JOINT_INDEPENDENCE)
    else:
        raise NotImplementedError

    for idx, outer_batch in enumerate(loader):
        if phase == 'train':
            # set the necessary train and eval modes for the model; this is done at the start because the model will be set to eval after the train loop

            if not pred_only and not critic_only and hparams.randomrestart > 0 and idx > 0 and (idx+1) % hparams.randomrestart == 0:
                # if pred only no critic related updates.
                # reinitialize critic model, z_representation, and critic model optimizers
                replace_model = create_model(hparams)
                replace_optimizers = configure_optimizers(
                    replace_model, hparams)
                model["critic_model"] = replace_model["critic_model"]
                model["z_representation"] = replace_model["z_representation"]
                optimizers[2] = replace_optimizers[2]

            if critic_only:
                
                EVAL_PRED_TRAIN_CRITIC(model)
                assert not model["x_representation"].training
                assert model["z_representation"].training

                assert len(optimizers) == 1, len(optimizers)
                opt_phi = optimizers[0]

                _ = _critic_train_helper(opt_phi, model, outer_batch, hparams, phase, device)
                # opt_phi.zero_grad()
                # metrics = compute_joint_independence_loss(
                #     x, y, w, model=model)
                # loss = metrics["critic_loss"]
                # loss.backward()
                # opt_phi.step()
            else:
                TRAIN_MODELS(model)

                if warmup_critic_flag and idx == 0:
                    # only done before iteration 1
                    print("WARMING UP CRITIC")
                    _, _, opt_phi = optimizers
                    # setting the pred model to eval and critic/z_rep to train. so batch norm doesn't change for the pred
                    EVAL_PRED_TRAIN_CRITIC(model)

                    # THE INNER LOOP WITH CRITIC TRAINING. PHI PARAMETER UPDATES OF SIZE TOTAL_BATCHES
                    iter_loader = iter(loader_copy)
                    total_batches = int(len(loader_copy)*hparams.frac_phi_steps)

                    for batch_idx in range(total_batches):
                        assert not critic_only, "SHOULD NOT DO THIS LOOP WHEN CRITIC ONLY. THAT IS SEPARATE CONDITION."
                        try:
                            batch = next(iter_loader)
                        except StopIteration:
                            iter_loader = iter(loader_copy)
                            batch = next(iter_loader)
                        
                        phi_metrics = _critic_train_helper(opt_phi, model, batch, hparams, phase, device)

                    assert not model['x_representation'].training
                # this is the update for the pred model
                TRAIN_PRED_EVAL_CRITIC(model)
                assert model['x_representation'].training
                assert not model['z_representation'].training

                # while updating the pred model, the critic model are set to eval because we do not want batch norm to change during the forward pass.
                model["z_representation"].eval()
                model["critic_model"].eval()

                # THE OUTER LOOP WITH REPRESENTATION AND PRED MODEL LEARNING. PHI PARAMETER UPDATES OF SIZE TOTAL_BATCHES
                _pred_train_helper(optimizers, model, outer_batch, hparams, phase, device, pred_only)

                _, _, opt_phi = optimizers

                # INNER LOOP; updates the critic model
                if not pred_only:
                    # setting the pred model to eval so batch norm doesn't change.
                    EVAL_PRED_TRAIN_CRITIC(model)

                    # THE INNER LOOP WITH CRITIC TRAINING. PHI PARAMETER UPDATES OF SIZE TOTAL_BATCHES
                    iter_loader = iter(loader_copy)
                    total_batches = int(len(loader_copy)*hparams.frac_phi_steps)

                    for batch_idx in range(total_batches):
                        assert not critic_only, "SHOULD NOT DO THIS LOOP WHEN CRITIC ONLY. THAT IS SEPARATE CONDITION."
                        try:
                            batch = next(iter_loader)
                        except StopIteration:
                            iter_loader = iter(loader_copy)
                            batch = next(iter_loader)
                        
                        phi_metrics = _critic_train_helper(opt_phi, model, batch, hparams, phase, device)


                        if hparams.debug > 2:
                            print(batch_idx)
                            utils.print_dict(phi_metrics)

        EVAL_MODELS(model) # set model to eval mode. these model are set to train at the start of an iteration during training above.
        with torch.no_grad():
            # this is just evaluation
            x, y, w = process_batch(
                outer_batch, hparams, device, phase="val", return_weights=hparams.METERS == "weighted")
            metrics = compute_joint_independence_loss(
                x, y, w, model, pred_only=pred_only, normalize=False)
            meters.update(metrics, N=x.shape[0])

            if hparams.debug > 2 and idx % hparams.print_interval == 0:
                normalized_metrics = compute_joint_independence_loss( x, y, w, model, pred_only=pred_only, normalize=True)
                print("====================")
                print(" BATCH {} with LAMBDA = {:.3f}".format(
                    idx, hparams.current_lambda))
                utils.print_dict(normalized_metrics)

    del loader_copy

    return meters.compile_result()


def generator_forward(y, z, model):
    # this is only for
    yz_cat = torch.cat([y.view(-1, 1), z], dim=1)
    mu = model["mu"].forward(yz_cat)
    sigma = model["sigma"].forward(yz_cat)

    return mu, sigma


def one_erm_loop(phase, loader, model, hparams, optimizers, critic_only=False, warmup_critic_flag=False):
    assert not critic_only, "This is a dummy variable used to standardize function calls"
    assert not warmup_critic_flag, "This is a dummy variable used to standardize function calls"
    # model is dict with one 1 thing
    # x_representation
    # pred_layer
    device = hparams.device

    metrics = {"pred_loss": 0, "acc": 0}

    meters = AvgMultiMeter(metrics)

    for idx, batch in enumerate(loader):
        # print(idx)
        # print(phase)
        if phase != 'train':
            model["x_representation"].eval()
            model["pred_layer"].eval()
        else:
            model["x_representation"].train()
            model["pred_layer"].train()

            opt_theta = optimizers[0]

            x, y, _ = process_erm_batch(batch, hparams, device, phase=phase)
            # to get the same performance as the group DRO people.
            opt_theta.zero_grad()
            metrics = compute_erm_loss(
                x, y, model=model, normalize=hparams.dataset in waterbirds_list)
            metrics['pred_loss'].backward()
            opt_theta.step()

        with torch.no_grad():
            x, y, _ = process_erm_batch(batch, hparams, device, phase=phase)
            metrics = compute_erm_loss(x, y, model)
            meters.update(metrics, y.shape[0])
    return meters.compile_result()


def predict_proba(loader, model, hparams):
    # model is dict with one 1 thing
    # x_representation
    # pred_layer
    device = hparams.device
    # idx_list = []
    proba_list = []

    for idx, batch in enumerate(loader):
        x, y, idx = process_erm_batch(batch, hparams, device, phase='val')
        y_pred_g_z = log_ratio_forward_pass_ermonly(model, x, y)
        # cross entropy gives NLL, exp of -NLL is probability predicted by the model
        proba_vec = (F.cross_entropy(y_pred_g_z, y.long().view(-1),
                     reduction='none').view(-1)*-1).exp()
        proba_list.append(proba_vec)
        # idx_list.append(idx)

    return utils.concat(proba_list)  # , utils.concat(idx_list)


ce_loss = torch.nn.CrossEntropyLoss(reduction='none')


def compute_erm_loss(x, y, model, normalize=False):

    # log ratio acc
    y_pred_g_rx = log_ratio_forward_pass_ermonly(model, x, y)

    acc_vec = acc_func(y_pred_g_rx, y)
    y_g_rx_loss_vec = ce_loss(y_pred_g_rx, y.long().view(-1)).view(-1)

    metrics = {
        "pred_loss": torch.sum(y_g_rx_loss_vec),
        "acc": torch.sum(acc_vec),
    }

    if normalize:
        return {"pred_loss": torch.mean(y_g_rx_loss_vec), "acc": torch.mean(acc_vec)}

    return metrics

def get_meters_to_use(objective, key, nr_strategy, train_flag):
    assert objective in ['erm', 'dist', 'critic_only']
    assert key in ['train', 'tr_fix', 'test', 'eval', 'val', "unbal_val"]
    assert nr_strategy in ['weight', 'upsample']
    assert type(train_flag) == bool

    if key == "unbal_val":
        meters_to_use = 'plain_avg'
    elif objective in ["dist", "critic_only"]:
        if key == "test" or key == "eval":
            meters_to_use = "plain_avg"
        elif nr_strategy == "upsample" and train_flag:
            meters_to_use = "plain_avg"
        else:
            meters_to_use = "weighted"
    else:
        # this is the default in erm_loop, so this setting is superfluous
        meters_to_use = "plain_avg"

    return meters_to_use


def run_one_epoch(_RUN_LOOP, args_bunch, loaders, model, optimizers, phase="train", objective="erm", epoch=0):
    # helps us to know the type of _RUN_LOOP to determine the METERS arg.
    assert objective in ['erm', 'dist', 'critic_only']
    critic_only_flag = objective == 'critic_only'  # only affects training, not val
    results_dict = {}
    for key, loader in loaders.items():
        # print(key)
        TRAIN_CHECK = key == "train" and phase == "train"
        torch.set_grad_enabled(TRAIN_CHECK)
        local_phase = "train" if TRAIN_CHECK else "val"
        local_optimizers = optimizers if TRAIN_CHECK else None
        # the following meters_to_use is just for the joint independence loop
        # meters to use determines decides between weighting the loss or just doing the plain average
        meters_to_use = get_meters_to_use(objective, key, args_bunch.nr_strategy, train_flag=TRAIN_CHECK)
        args_bunch.update({'METERS': meters_to_use})

        results_dict[key] = _RUN_LOOP(
            local_phase, loader, model, args_bunch, local_optimizers, 
            critic_only=critic_only_flag, warmup_critic_flag=args_bunch.warmup_critic and epoch==0 and objective=='dist')
    
    return results_dict


def fit_model(args_bunch, epochs, loaders, objective, writer, saver, loaded_model=None):
    assert objective in ['erm', 'dist', 'critic_only'], f"{objective} unknown."

    # logging and saving
    _writer = writer  # create in main loop
    _saver = saver  # create in main loop

    # model and optimizer creation
    if objective == "erm":
        # this is used only for nuisance randomization
        assert args_bunch.weight_model
        _model = create_erm_model(args_bunch)
        _optimizers = configure_erm_optimizers(_model, args_bunch)
        _RUN_LOOP = one_erm_loop
    else:
        # objective is dist
        assert not args_bunch.weight_model
        assert not args_bunch.conditional, "FUNCTIONALITY NOT CHECKED."
        _RUN_LOOP = one_joint_independence_loop
        _REGULARIZER_LOSS = "info_loss"
        if objective == "dist":
            _model = create_model(args_bunch)
            _optimizers = configure_optimizers(_model, args_bunch)
        else:
            # this is critic only, so model is loaded
            assert loaded_model is not None
            _model = copy.deepcopy(loaded_model)
            replace_model = create_model(args_bunch)
            _model["critic_model"] = replace_model["critic_model"]
            _model["z_representation"] = replace_model["z_representation"]
            _optimizers = configure_optimizers(_model, args_bunch)
            # delete the first two optimizers; only the last one matters.
            del _optimizers[0]
            del _optimizers[0]
            assert len(_optimizers) == 1

    EVAL_MODELS(_model)

    for epoch in range(epochs):

        epoch_results = run_one_epoch(
            _RUN_LOOP,
            args_bunch,
            loaders,
            _model,
            _optimizers,
            phase="train",
            objective=objective,
            epoch=epoch
        )

        if args_bunch.verbose >= 2:
            print(
                " EPOCH = {} ========================================================".format(epoch))

        for key, result in epoch_results.items():
            _writer.add_scalars(key, result, global_step=epoch)

            if args_bunch.verbose >= 2:
                print(f" DEBUG, {key}", end="\t")
                utils.print_dict(result)

        val_metrics = epoch_results['val']

        # this conditional computes different val_loss for different objecives
        if objective == "erm":
            val_loss = val_metrics['pred_loss']
        else:
            # objective is "nurd"
            # first update current_lambda; training takes current lambda
            # second, ensure non-negative and less than max_lambda_

            if args_bunch.anneal > 1e-3:
                assert False
                args_bunch.current_lambda = args_bunch.current_lambda + args_bunch.anneal
                args_bunch.current_lambda = max(
                    0, min(args_bunch.current_lambda, args_bunch.max_lambda_))
                if args_bunch.verbose == 4:
                    print(" UPDATED LAMBDA = {:.3f}".format(
                        args_bunch.current_lambda))

            if objective == "critic_only":
                val_loss = val_metrics['critic_loss']
            else:
                assert val_metrics[_REGULARIZER_LOSS] >= 0, val_metrics[_REGULARIZER_LOSS]
                val_loss = args_bunch.max_lambda_ * \
                    val_metrics[_REGULARIZER_LOSS] + val_metrics['pred_loss']

        EVAL_MODELS(_model)
        _saver.maybe_save(epoch, _model, metric_val=val_loss,
                          print_flag=args_bunch.debug >= 1)
        
        # if args_bunch.nr_strategy == "weight":
        # if abs(epoch_results['train']['pred_loss'] - epoch_results['tr_fix']['pred_loss']) > 1e-4:
        # logging.warning("DEBUG:weighted train and tr_fix do not match : {:.5f}/{:.5f}".format(epoch_results['train']['pred_loss'], epoch_results['tr_fix']['pred_loss'] ))

    # load and set model to eval
    best_model = _saver.load_best(map_location=args_bunch.device,
                                  objective="dist" if objective == "critic_only" else objective)

    EVAL_MODELS(best_model)

    # print post training results
    print("")
    print(" DONE WITH FITTING MODELS FOR OBJECTIVE {}".format(objective))

    best_model_results = run_one_epoch(_RUN_LOOP, args_bunch, loaders,
                                   best_model, optimizers=None, phase="val", objective=objective)
    utils.print_dict_of_dicts(best_model_results)

    # cast to CPU
    CPU_MODELS(_model)

    _writer.close()
    return best_model, best_model_results