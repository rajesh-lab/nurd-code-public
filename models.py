# models.py
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

import sys
from os.path import dirname
filepath = os.path.abspath(__file__)
erm_dir = dirname(filepath)
parentdir = dirname(erm_dir)
sys.path.append(parentdir)
from utils import waterbirds_list

class TopLayer(nn.Module):
    def __init__(self, in_size, out_size, nonlin=False):
        super().__init__()
        self.nonlin = nonlin

        if self.nonlin:
            mid_size = 16
            # self.bn1 = nn.BatchNorm1d(in_size)
            self.top = nn.Linear(in_size, mid_size, bias=False)
            # self.bn2 = nn.BatchNorm1d(mid_size)
            self.top2 = nn.Linear(mid_size, mid_size, bias=False)
            self.top3 = nn.Linear(mid_size, out_size, bias=False)
        else:
            self.top = nn.Linear(in_size, out_size) #, bias=False)

    def forward(self, x):
        if self.nonlin:
            # return self.top3(F.relu(
            # x = self.top2(F.relu(self.top1(self.bn1(x) )))
            x = self.top2(F.relu(self.top(x) ))
            # x = self.bn2(x)
            return self.top3(F.relu(x))

        return self.top(x)


        
class TwoLayerMLP(nn.Module):
    def __init__(self, in_size, out_size=2, mid_size=64):
        super().__init__()
        self.fc1 = nn.Linear(in_size, mid_size)
        self.fc2 = nn.Linear(mid_size, out_size)

    def forward(self, x):
        x = F.relu(  self.fc1(x) )
        return self.fc2(x)

# class TwoLayerMLP(nn.Module):
#     def __init__(self, in_size, out_size=2, mid_size=64):
#         super().__init__()
#         self.fc1 = nn.Linear(in_size, mid_size)
#         self.fc2 = nn.Linear(mid_size, out_size)

#     def forward(self, x):
#         x = F.relu(  self.conv1(x) )
#         return self.fc2(x)

class SmallFeaturizer1d(nn.Module):
    def __init__(self, in_size, out_size=64):
        super().__init__()
        self.fc1 = nn.Linear(in_size, out_size)
        # typically 2 more layers are added after the featurizer.
        # self.bn1 = nn.BatchNorm1d(out_size)

    def forward(self, x):
        # x = self.bn1( 
        x = F.relu(  self.fc1(x) )
        return x

class SmallFeaturizer(nn.Module):
    def __init__(self, in_channels, weight_model=True, _for_critic_model=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.dist_model=not weight_model
        self.critic_model=_for_critic_model
        DROPOUT_RATE = 0.5
        if not self.critic_model:
            self.dropout1 = nn.Dropout(DROPOUT_RATE)
            self.dropout2 = nn.Dropout(DROPOUT_RATE)
            self.dropout3 = nn.Dropout(DROPOUT_RATE)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        if not self.dist_model:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        else:
            x = self.pool(self.bn1(F.relu(self.conv1(x))))
            x = self.dropout1(x) if not self.critic_model else x # dropout only for the dist model
            x = self.pool(self.bn2(F.relu(self.conv2(x))))
            x = self.dropout2(x) if not self.critic_model else x # dropout only for the dist model
            x = self.pool(self.bn3(F.relu(self.conv3(x))))
        return x

class MnistFeaturizer(nn.Module):
    def __init__(self):
        super(MnistFeaturizer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

def process_model_type(model_type, model_stage, _for_critic_model=False):
    # print("USING MODEL = ", model_type)
    if model_type == "resnet":
        network = torchvision.models.resnet18(pretrained=False)
        network.conv1 = torch.nn.Conv1d(
            1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    elif model_type == "resnet_color":
        network = torchvision.models.resnet18(pretrained=True)
    elif model_type == "resnet_color_init":
        network = torchvision.models.resnet18(pretrained=False)
    elif model_type == "resnet50_color":
        network = torchvision.models.resnet50(pretrained=True)
    elif model_type == "resnet50_color_init":
        network = torchvision.models.resnet50(pretrained=False)
    elif model_type == "densenet_color":
        network = torchvision.models.densenet121(pretrained=True)
    elif model_type == "densenet_color_init":
        network = torchvision.models.densenet121(pretrained=False)
    elif model_type == "small":
        network = SmallFeaturizer(in_channels=1, weight_model=model_stage=="weight_model", _for_critic_model=_for_critic_model)
    elif model_type == "cmnist":
        network = MnistFeaturizer()
    elif model_type == "small_color":
        network = SmallFeaturizer(in_channels=3, weight_model=model_stage=="weight_model", _for_critic_model=_for_critic_model)
    elif model_type == "synthetic1d":
        # this assumes synthetic data of this type
        assert False, "synthetic 1d should not be called here"
    else:
        assert False, "BRO MODEL NOT DEFNIED  HOW DARE YOU?"
    return network


class FFRepresentation(nn.Module):
    def __init__(self, hparams, _transform_type, mid_dim):
        super().__init__()

        self.transform_func = construct_transform(_transform_type, N_CHANNELS=None, SIDE=hparams.img_side)
        self.type = hparams.pred_model_type
        if self.type == "linear":
            self.lin_rep = torch.nn.Linear(3, hparams.out_dim, bias=False)
        else:
            self.network = SmallFeaturizer1d(in_size=3, out_size=mid_dim)
            self.l1 = torch.nn.Linear(mid_dim, hparams.out_dim, bias=False)

    def forward(self, x, save_transformed=False, h=None):
        x = self.transform_func(x)
        if save_transformed:
            print(x)
        if self.type == "linear":
            return self.lin_rep(x) # l1 maps this to u
        else:
            x = torch.relu(self.network(x).view(x.shape[0], -1))
            return self.l1(x) # l1 maps this to u


class ImageRepresentation(nn.Module):
    def __init__(self, hparams, _transform_type, mid_dim=None, _for_critic_model=False):
        super().__init__()
        assert mid_dim is None, "Not used"

        assert hparams.model_stage in ['weight_model', 'pred_model'], hparams.model_stage

        self.debug = hparams.verbose == 19 # specify 20 to break and save images at weight, 19 for pred stage

        self.hparams = hparams
        # self.transform_func = construct_transform(_transform_type, N_CHANNELS=3 if hparams.dataset=='cmnist' else 1, SIDE=hparams.img_side, hparams=hparams)
        self.transform_func = construct_transform(_transform_type, N_CHANNELS=3 if hparams.dataset in ['cmnist'] + waterbirds_list else 1, SIDE=hparams.img_side, hparams=self.hparams)
        self.network = process_model_type(model_type=hparams.model_type, model_stage=hparams.model_stage, _for_critic_model=_for_critic_model)

        with torch.no_grad():
            if hparams.model_type == "cmnist":
                lin_dim = self.network(torch.randn( (1, 3, hparams.img_side, hparams.img_side))).view(-1).shape[0]
            elif hparams.model_type == "synthetic1d":
                assert False, "USE FFRepresentation"
            else:
                IN_CHANNELS = 3 if hparams.dataset in waterbirds_list or hparams.color else 1
                lin_dim = self.network(torch.randn((2, IN_CHANNELS, hparams.img_side, hparams.img_side))).view(2, -1).shape[1]
                # lin_dim = self.network(torch.randn((2, 1, hparams.img_side, hparams.img_side))).view(2, -1).shape[1]
        self.lin_dim = lin_dim 

        # print("CREATED LINEAR TOP LAYER OF SHAPE FROM {} TO {}".format(self.lin_dim, 1))
        self.l1 = torch.nn.Linear(self.lin_dim, hparams.out_dim, bias=False)

    def forward(self, x, save_transformed=False, h=None):
        if self.debug:
            print("SAVING BEFORe TRANSFORM")
            save_image(x, "./DEBUG_BEFORE_TRANSFORM.png",  nrow=8, normalize=True, scale_each=True)
        x = self.transform_func(x)
        # print(x.abs().max())
        if self.debug:
            print("SAVING AFTER TRANSFORM")
            save_image(x, "./DEBUG_AFTER_TRANSFORM.png",  nrow=8, normalize=True, scale_each=True)
            assert False
        x = torch.relu(self.network(x).view(x.shape[0], -1))
        return self.l1(x) # l1 maps this toua vector
