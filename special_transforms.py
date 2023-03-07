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
from torchvision.utils import save_image


class RandomCrop:
    """Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image. 
            Default is None, i.e no padding.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """
    
    def __init__(self, size, device='cpu'):
        self.size = size
        self.padding = None
        self.device = device
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        IN_SIZE=tensor.shape[-1]
        if self.padding is not None:
            padded = torch.zeros((tensor.size(0), tensor.size(1), tensor.size(2) + self.padding * 2, 
                                  tensor.size(3) + self.padding * 2), dtype=tensor.dtype, device=self.device)
            padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = tensor
        else:
            padded = tensor
        # print(padded.shape)
            
        h, w = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),), device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),), device=self.device)
            
        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded_zeros = torch.zeros_like(padded)
        padded_zeros[:, torch.arange(tensor.size(0))[:, None, None], rows[:, torch.arange(th)[:, None]], columns[:, None]] = padded[:, torch.arange(tensor.size(0))[:, None, None], rows[:, torch.arange(th)[:, None]], columns[:, None]]
        return padded_zeros.permute(1, 0, 2, 3)

def construct_transform(transform, N_CHANNELS, SIDE, hparams=None):
    if transform == "identity":
        return lambda x : x
    
    if hparams is not None:
        h = w = int(SIDE*hparams.border)//8

        margin_h = (SIDE-h)//2
        margin_w = (SIDE-w)//2
    
    if "gaussian_blur" in transform:
        def blur(x):
            return torchvision.transforms.functional.gaussian_blur(x, kernel_size=(hparams.blur_kernel,hparams.blur_kernel))
        return blur
        
    if "intensity" in transform:
        rho = hparams.percentile
        def _filter_max(x):
            N_CHANNELS = x.shape[1]
            _intensity = torch.sum(x, dim=1) # perform fft
            shape = _intensity.shape
            _intensity_reshaped = _intensity.view(shape[0], -1)
            _percentile = torch.quantile(_intensity_reshaped, rho, dim=1).unsqueeze(1)
            # if transform == "intensity_high":
            _intensity_mask = (_intensity_reshaped > _percentile).view(shape).unsqueeze(1)
            # else:
            #     _intensity_mask = (_intensity_reshaped < _percentile).view(shape).unsqueeze(1)
            _intensity_mask_nchannel = torch.cat([_intensity_mask]*N_CHANNELS, dim=1)
            x_clone = x.clone()
            x_clone[_intensity_mask_nchannel] = -2
            return x_clone

        return _filter_max

    if "pass" in transform:
        assert SIDE==224
        _h = _w = hparams.freq

        if "band" in transform:
            margin_h = (SIDE-_h)//4
            margin_w = (SIDE-_w)//4

            mask = torch.ones(1, N_CHANNELS, SIDE, SIDE)
            if _h > 0 and _w > 0:
                mask[:,:,margin_h : SIDE - margin_h, margin_w : SIDE - margin_w] = 0 # this is for band pass
                mask[:,:, - margin_h + SIDE//2 : margin_h + SIDE//2,  - margin_w + SIDE//2 : margin_w + SIDE//2] = 1 # this is for band pass
            if transform == "band_pass_keep" and _h > 0 and _w > 0:
                mask = 1 - mask # keep only the band, without this the band is thrown away
        else:
            margin_h = (SIDE-_h)//2
            margin_w = (SIDE-_w)//2

            mask = torch.ones(1, N_CHANNELS, SIDE, SIDE)
            mask[:,:,margin_h : _h + margin_h, margin_w : _w + margin_w] = 0 # this is for highpass
            if transform == "low_pass":
                mask = 1 - mask
        
        def _masker(x):
            mask_gpu = mask.to(x.device)
            return x*mask_gpu # this broadcasts over the batch dimension, sets hole to -1

        def _pass(x):
            _fft = torch.fft.fft2(x) # perform fft
            _fft_shifted = torch.fft.fftshift(_fft, dim=(-2,-1)) # center the 0 components
            _fft_shifted_highpass = _masker(_fft_shifted) # mask out the center
            _fft_highpass = torch.fft.ifftshift(_fft_shifted_highpass, dim=(-2,-1)) # undo the centering
            _x_highpass = torch.fft.ifft2(_fft_highpass).real # inverse fft
            return _x_highpass.to(x.device)

        return _pass
    
    if transform == "zero_nuisance":
        def zero_nuisance(x):
            return 0*x
        return zero_nuisance
    
    if transform == "hybrid":
        print("USING HYBRID")
        PS=hparams.patch_size
        def _pr_middle_transform_func(_batch_x):
            batch_x = _batch_x[:,:,margin_h : h + margin_h, margin_w : w + margin_w]
            bs = batch_x.shape[0]
            SIDE=batch_x.shape[-1]
            imgs_cut = batch_x.unfold(2, PS, PS).unfold(3, PS, PS)
            imgs_cut_correct_shape = imgs_cut.shape
            imgs_cut = imgs_cut.reshape(bs, N_CHANNELS, -1, PS, PS)
            imgs_cut = imgs_cut[:,:,torch.randperm(imgs_cut.shape[2]),:,:].reshape(imgs_cut_correct_shape)
            return imgs_cut.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(bs, N_CHANNELS, SIDE, SIDE)

        # assert False, (h,w)
        mask = torch.ones(1, N_CHANNELS, SIDE, SIDE)
        mask[:,:,margin_h : h + margin_h, margin_w : w + margin_w] = 0
        def _transform_func(x):
            mask_gpu = mask.to(x.device)
            middle_pr = _pr_middle_transform_func(x).to(x.device)
            x_return = x*mask_gpu
            x_return[:,:,margin_h : h + margin_h, margin_w : w + margin_w] = middle_pr
            # assert False, (x_return.shape, x.shape, middle_pr.shape)
            return x_return
        
        return _transform_func

    if transform == "HG":
        sigma=hparams.sigma
        print("USING HEAVY GAUSSIAN NUISANCE AUGMENTATION with sigma = ", sigma)
        def add_gaussian_noise(x):
            return x + sigma*torch.randn(x.shape).to(x.device)
        return add_gaussian_noise

    if transform == "crop":
        _size=hparams.patch_size
        
        # print("USING HEAVY GAUSSIAN NUISANCE AUGMENTATION with sigma = ", sigma)
        def random_crop(x):
            _transform = RandomCrop(size=_size, device=x.device)
            return _transform(x)
        return random_crop
    
    if transform == "synthetic_last":
        return lambda x: torch.cat([0*x[:,:2],x[:,2].view(-1,1)], dim=1)
    
    if transform == "synthetic_first_two":
        # return lambda x: x[:,:2]
        return lambda x: torch.cat( [x[:,:2], 0*x[:,2].view(-1,1)], dim=1)

    if transform == "PR":
        print("USING PATCH RANDOMIZATION NUISANCE AUGMENTATION")
        PS=hparams.patch_size
        def _transform_func(batch_x):
            bs = batch_x.shape[0]
            SIDE=batch_x.shape[-1]
            imgs_cut = batch_x.unfold(2, PS, PS).unfold(3, PS, PS)
            imgs_cut_correct_shape = imgs_cut.shape
            imgs_cut = imgs_cut.reshape(bs, N_CHANNELS, -1, PS, PS)
            imgs_cut = imgs_cut[:,:,torch.randperm(imgs_cut.shape[2]),:,:].reshape(imgs_cut_correct_shape)
            return imgs_cut.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(bs, N_CHANNELS, SIDE, SIDE)
        
        return _transform_func

    # if SIDE==224:
    #     h = SIDE*7//8 # 32 SIDE gives h,w = 24 
    #     w = SIDE*7//8 # 32 SIDE gives h,w = 24
    # elif SIDE==32:
    #     h = 24
    #     w = 24
    # else:
    #     assert False, "you sure you want holemask with this data set?"
    if transform == "holemask":
        mask = torch.ones(1, N_CHANNELS, SIDE, SIDE)
        mask[:,:,margin_h : h + margin_h, margin_w : w + margin_w] = 0
        def _transform_func(x):
            mask_gpu = mask.to(x.device)
            return x*mask_gpu - (1-mask_gpu) # this broadcasts over the batch dimension, sets hole to -1

    elif transform == "side_and_top_mask":
        mask = torch.ones(1, N_CHANNELS, SIDE, SIDE)
        mask[:,:,margin_h : h + margin_h, margin_w : w + margin_w] = 0
        mask[:,:, w + margin_w:, margin_h : h + margin_h] = 0
        mask[:,:,: margin_w,:] = 0
        def _transform_func(x):
            mask_gpu = mask.to(x.device)
            return x*mask_gpu - (1-mask_gpu) # this broadcasts over the batch dimension, sets hole to -1

    elif transform == "onlycenter":
        # reverse mask of the holemask
        mask = torch.zeros(1, 1, SIDE, SIDE)
        mask[:,:,margin_h : h + margin_h, margin_w : w + margin_w] = 1
        def _transform_func(x):
            # print("PREDICTION FROM CENTER.")
            mask_gpu = mask.to(x.device)
            return x*mask_gpu - (1-mask_gpu)
    
    elif transform == "cmnist_color":
        def _transform_func(batch_x):
            c = torch.max(batch_x + 1, dim=2)[0]
            c = c.max(dim=2)[0]
            c = c.argmax(dim=1)
            batch_c = c.view(-1, 1, 1, 1)
            assert batch_c.shape == (batch_x.shape[0], 1, 1, 1), 'WUT'

            ret_img = batch_x*0 + (2*batch_c - 1)

            return ret_img

    elif transform == "colorize":
        def _transform_func(batch_x):
            batch_x_colorized = 0*batch_x
            for i in range(batch_x.shape[0]):
                which_color = torch.randint(low=0,high=3,size=(1,)) # size=(batch_x.shape[0], ) )
                # print(which_color)
                batch_x_colorized[i, which_color] = batch_x[i].sum(dim=0)
            save_image(batch_x_colorized, "colorized_mnist.png",  nrow=8, normalize=True, scale_each=True)
            # assert False, (batch_x.shape, batch_x.min(), batch_x.max(), which_color.shape)
            return batch_x_colorized
    
    elif transform[:7] == "toprows":
        try:
            n_rows = int(transform[8:])
            assert n_rows < SIDE
            print('N_ROWS = ', n_rows)
        except:
            assert False, "{} is not only int; please enter strings like toprows_2".format(transform[8:])

        def _transform_func(x, y):
            x[:, :, n_rows:, :] = -1 # -1 because the images lowest value is this.
            return x
        for batch in train_loader:
            x = batch[0]
            samples = _transform_func(x[:16], None)[:16]
            save_image(samples, "transformed_{}.png".format(transform),  nrow=8, normalize=True, scale_each=True)
            break
        print(' ---- TOPROWS ONLY TRANSFORM WITH {} rows'.format(n_rows))
        print(' ---- TOPROWS ONLY TRANSFORM WITH {} rows'.format(n_rows))

    elif transform[:7] == "botrows":
        try:
            n_rows = int(transform[8:])
            assert n_rows < SIDE

            print('N_ROWS = ', n_rows)
        except:
            assert False, "{} is not only int; please enter strings like BOTrows_2".format(transform[8:])

        def _transform_func(x, y):
            x[:, :, :x.shape[2] - n_rows, :] = -1 # -1 because the images lowest value is this.
            return x

        # for batch in train_loader:
        #     x = batch[0]
        #     samples = _transform_func(x[:16], None)[:16]
        #     save_image(samples, "transformed_{}.png".format(transform),  nrow=8, normalize=True, scale_each=True)
        #     break
        print(' ---- BOTROWS ONLY TRANSFORM WITH {} rows'.format(n_rows))
        print(' ---- BOTROWS ONLY TRANSFORM WITH {} rows'.format(n_rows))


    elif 'COVIDomaly' in transform:
        assert False

        assert not nurd, "NURD for autoencoder not implemented"
        print(' ---- LOADING AUTOENCODER')
        print(' ---- LOADING AUTOENCODER')

        sys.path.append("/scratch/apm470/nuisance-orthogonal-prediction/code/nrd-xray/COVIDomaly/")
        from utils.NeuralNet import Generator

        tmodel = Generator(height=SIDE, width=SIDE, channel=1,  
                    device=device, ngpu=1,
                    ksize=5, z_dim=args.z_dim, learning_rate=0.0)
        tmodel.load_state_dict(torch.load(transform, map_location=device))

        _transform_func= lambda x, y : tmodel(x)

    elif transform[:10] == "discretize":            
        # assert not nurd, "NURD for discretize does not make sense"

        try:
            n_categories = int(transform[11:])
            print('N_CAT = ', n_categories)
        except:
            assert False, "{} is not only int; please enter strings like discretize_16".format(transform[11:])

        def _transform_func(x, y):
            # [-1, 1] ---> [0, k]
            x_unit = 0.4999*(0.5 + x) # [-1, 1] ---> [0, 1]
            return -1 + 2*(torch.floor(x_unit*n_categories)/n_categories) # [0, 1) ---> [0, k-1] ---> [0, 1] ----> [-1, 1])

        print(' ---- LOADING DISCRETIZATION TRANSFORM WITH {} CATEGORIES'.format(n_categories))
        print(' ---- LOADING DISCRETIZATION TRANSFORM WITH {} CATEGORIES'.format(n_categories))

    elif "pixelcnn-pytorch" in transform:
        assert False
        
        print(' ---- LOADING PIXELCNN TRANSFORM')
        print(' ---- LOADING PIXELCNN TRANSFORM')

                    
        sys.path.append("/scratch/apm470/nuisance-orthogonal-prediction/code/nrd-xray/pixelcnn-pytorch/")
        from models_pcnn import ConditionalPixelCNN
        from utils import generate_samples, generate_inpainted

        sys.path.append("/scratch/apm470/generative-inpainting-pytorch/utils/")
        from tools import bbox2mask

        config = torch.load("/".join(transform.split('/')[:-1]) + "/config.pt")
        img_shape, n_classes = [config.img_side, config.img_side], 2
        n_categories = config.n_categories
        no_channels, convolution_filters = 1, config.n_conv
        cpixelcnn = ConditionalPixelCNN(no_channels, convolution_filters, num_classes=n_classes+2, n_categories=n_categories).cuda()
        cpixelcnn.load_state_dict(torch.load(transform))

        side = 32
        h = 24
        w = 28
        bboxes = torch.tensor([(int((side-h)/2), int((side-w)/2), h, w)]*hparams['batch_size'], dtype=torch.int64) # random_bbox(config, batch_size=hparams['batch_size'])
        mask = bbox2mask(bboxes, 32, 32, 2, 2).cuda()
        
        def one_hot_labels(labels):
            labels_oh = torch.zeros((labels.shape[0], n_classes))
            labels_oh[labels>0, 1] = 1
            labels_oh[labels<1, 0] = 1
            return labels_oh

        if nurd:
            _transform_func = lambda x, y : generate_inpainted(one_hot_labels(y.view(-1).long()), one_hot_labels(y.view(-1).long()), cpixelcnn, mask, x, n_categories=n_categories, image_shape=img_shape, no_channels=no_channels).cuda()
            for batch in train_loader:
                x = batch[0]
                halfsize = x.shape[0] // 2
                y = torch.Tensor([1]*halfsize + [0]*halfsize).cuda()
                samples = _transform_func(x.cuda(), y)[:32]
                save_image(samples, "generation_diff_top2rows_y1_NURD.png",  nrow=8, normalize=True, scale_each=True)
                break
                
        else:
            _transform_func = lambda x, y : generate_samples(one_hot_labels(y.view(-1).long()), one_hot_labels(y.view(-1).long()), cpixelcnn, n_categories=n_categories, image_shape=img_shape, no_channels=no_channels).cuda()

            y = torch.Tensor([1]*16 + [0]*16).cuda()
            samples = _transform_func(None, y)
            save_image(samples, "generation_diff_top2rows_y1.png",  nrow=8, normalize=True, scale_each=True)
        # assert False
    else:
        assert False, 'BRO, dont know how to handle gan'

    return _transform_func
