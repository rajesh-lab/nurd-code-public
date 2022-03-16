import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import time

import sys

def filter_function(batch_x):
    _batch_x = 0.5*(1 + batch_x)
    print(_batch_x.min(), _batch_x.max())
    border_check = ((_batch_x[:, :, :, -2:].mean(dim=-1).mean(dim=-1)  < 1e-2) 
            + (_batch_x[:, :, -2:, :].mean(dim=-1).mean(dim=-1)   < 1e-2) 
            + (_batch_x[:, :, :2, :].mean(dim=-1).mean(dim=-1)   < 1e-2) 
            + (_batch_x[:, :, :, :2].mean(dim=-1).mean(dim=-1)   < 1e-2)
            + (_batch_x[:, :, :, -2:].mean(dim=-1).mean(dim=-1)  > 0.99) 
            + (_batch_x[:, :, -2:, :].mean(dim=-1).mean(dim=-1)  > 0.99) 
            + (_batch_x[:, :, :2, :].mean(dim=-1).mean(dim=-1)   > 0.99) 
            + (_batch_x[:, :, :, :2].mean(dim=-1).mean(dim=-1)   > 0.99)
           ).view(-1)
    
#     stat_across_pixels = (_batch_x.view(batch_x.shape[0], -1) > 0.9).float().mean(dim=-1)
    stat_across_pixels = (_batch_x.std(dim=-1).mean(dim=-1).view(-1) - _batch_x.std(dim=2).mean(dim=-1).view(-1)).abs()# (batch_x.shape[0], -1), dim=1)
    # print(stat_across_pixels)
    stat_check = stat_across_pixels<0.06
#     print(stat_across_pixels)
    
#     return stat_check > 0
    return ( border_check + stat_check )> 0

side = 32
from datasets import MimicCXRDataset        
mimic = MimicCXRDataset(None,
                        mode='train', upsample=False,
                        subset=True,
                        input_shape=[1, side, side])

mimic._mode = 'output'
mimic_loader = torch.utils.data.DataLoader(mimic, batch_size=200, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)

image_paths_to_keep_train = []
totlen = len(mimic)
time_now = time.time()
    
print("STARTING TRAINING KEEP")
print("STARTING TRAINING KEEP")
print("STARTING TRAINING KEEP")
print("STARTING TRAINING KEEP")
print("STARTING TRAINING KEEP")


count = 0
kept_count = 0
for ind, batch in enumerate(mimic_loader):
    x, names = batch
    
    keep = np.logical_not(filter_function(x).numpy())
    
    image_paths_to_keep_train.append(np.array(names)[keep])
    
    count += x.shape[0]
    kept_count += keep.sum()
    
    
    # if count % 100 == 0:
    print("LAPSED {:.3f} s".format(time.time() - time_now))
    time_now = time.time()
    print("done with {}/{}".format(count, totlen))
    print("kept with {}/{}".format(kept_count , totlen))
        
with open("./mimic_keep_bigcut_train.npz", 'wb') as f:
    arr = np.concatenate(image_paths_to_keep_train, axis=0)
    np.save(f, arr)
    
    
print("DONE WITH TRAINING KEEP")
print("DONE WITH TRAINING KEEP")
print("DONE WITH TRAINING KEEP")
print("DONE WITH TRAINING KEEP")
print("DONE WITH TRAINING KEEP")


mimic = MimicCXRDataset(None,
                        mode='val', upsample=False,
                        subset=True,
                        input_shape=[1, side, side])

mimic._mode = 'output'
mimic_loader = torch.utils.data.DataLoader(mimic, batch_size=200, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)

image_paths_to_keep_val = []
totlen = len(mimic)
time_now = time.time()
count = 0
for ind, batch in enumerate(mimic_loader):
    x, names = batch
    
    keep = np.logical_not(filter_function(x).numpy())
    
    image_paths_to_keep_val.append(np.array(names)[keep])
    
    count += x.shape[0]
    kept_count += keep.sum()
    
    # if count % 100 == 0:
    print("LAPSED {:.3f} s".format(time.time() - time_now))
    time_now = time.time()
    print("done with {}/{}".format(count, totlen))
    print("kept with {}/{}".format(kept_count , totlen))

    
print("DONE WITH VALIDATION KEEP")
print("DONE WITH VALIDATION KEEP")
print("DONE WITH VALIDATION KEEP")
print("DONE WITH VALIDATION KEEP")
print("DONE WITH VALIDATION KEEP")

#     break
with open("./mimic_keep_bigcut_val.npz", 'wb') as f:
    arr = np.concatenate(image_paths_to_keep_val, axis=0)
    np.save(f, arr)