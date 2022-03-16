import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models_attrs import model_attributes
from torch.utils.data import Dataset, Subset

class ConfounderDataset(Dataset):
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 model_type=None, augment_data=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.filename_array)
    
    def get_image(self, img_filename, idx):
        img = Image.open(img_filename).convert('RGB')
        # Figure out split and transform accordingly
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
            self.eval_transform):
            img = self.eval_transform(img)
        # Flatten if needed
        if model_attributes[self.model_type]['flatten']:
            assert img.dim()==3
            img = img.view(-1)
        return img

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        if model_attributes[self.model_type]['feature_type']=='precomputed':
            x = self.features_mat[idx, :]
        else:
            img_filename = os.path.join(
                self.data_dir,
                self.filename_array[idx]
            )
            x = self.get_image(img_filename, idx)
            if self.counterfactual:
                counterfactual_img_filename = os.path.join(
                self.data_dir,
                self.counterfactual_filename_array[idx]
            )
                x_counter = self.get_image(counterfactual_img_filename, idx)
        
                # print(x.shape, x_counter.shape)
                return x, y, g - 2*y, x_counter, idx         
        return x,y,g - 2*y, idx # in NURD this is becomes x,y,h

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ('train','val','test'), split+' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac<1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            
            if split == 'test':
                num_to_retain = int(np.round(float(len(indices))))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            print(np.unique(self.group_array[indices], return_counts=True))
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes) # second bit and first bit.
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name
    
    def group_yc_val(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes) # second bit and first bit.
        c = group_idx % (self.n_groups//self.n_classes)

        print(" y,c print ", y,c)

        return y,c

        # group_name = f'{self.target_name} = {int(y)}'
        # bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        # for attr_idx, attr_name in enumerate(self.confounder_names):
            # group_name += f', {attr_name} = {bin_str[attr_idx]}'
        # return group_name

