# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from os.path import dirname, abspath
import os
import torch
from easydict import EasyDict as edict
import json
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from dataset_utils import transform, GetTransforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    # "Debug28",
    # "Debug224",
    # Small images
    "ColoredMNIST",
    # "RotatedMNIST",
    # Big images
    # "VLCS",
    # "PACS",
    # "OfficeHome",
    # "TerraIncognita",
    # "DomainNet",
    # "SVIRO",
    'chestXR',
    'JointDataset'
]

all_diseases = ['Atelectasis', 'Cardiomegaly',
                'Consolidation', 'Edema', 'Pneumonia']
diseases = ['Pneumonia']
# csv_path = '/scratch/wz727/chestXR/data/labels/'
csv_path = '/home/apm470/xray_labels/'

domainbed_path = dirname(abspath(__file__))
# print('In folder = {}'.format(domainbed_path))
KEEP_COUNT_MAX = 15000

class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class ChestDataset(Dataset):
    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        COLOR_FLAG = self.input_shape[0]==3
        image = cv2.imread(self._image_paths[idx], int(COLOR_FLAG))
        # print(self._image_paths[idx])
        # print('loading idx {}'.format(idx))
        image = Image.fromarray(image)
        # image = transform(image.transpose((2, 0, 1)), self.cfg)
        # try:
        #     image = Image.fromarray(image)
        # except:
        #     raise Exception('None image path: {}'.format(self._image_paths[idx]))
        # resize = transforms.Resize()
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if COLOR_FLAG else transforms.Normalize((0.5), (0.5)),
             transforms.Resize(self.input_shape[1:]),
            ])
        image = transform(image)


        # assert False, (image.shape)
        
        if self.aug_transform and self._mode == 'train':
            assert False
            image = GetTransforms(image, type="Aug")
        
        labels = torch.Tensor(np.array([self._labels[idx]]).astype(np.float32))
        path = self._image_paths[idx]
        if self._mode == 'train' or self._mode == 'val' or self._mode == 'test' or self._mode == 'dry':
            if self._hosp is not None:
                hosp = torch.Tensor(np.array([self._hosp[idx]]).astype(np.float32))
                return (image, labels, hosp)
            else:
                return (image, labels)
        elif self._mode == 'output':
            return (image, path)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))

    def upsample(self):
        assert self._labels.shape[1] == 1
        # ratio = (len(self._labels) - self._labels.sum(axis=0)
        #          ) // self._labels.sum(axis=0) - 1
        # ratio = ratio[self._idx][0]
        # assert False, (self._idx, self._labels.shape, ratio)
        indices = np.arange(len(self._labels))
        disease_indices = indices[(self._labels[:, self._idx] > 0).ravel()]
        healthy_indices = indices[(self._labels[:, self._idx] < 1).ravel()]
        disease_count = disease_indices.shape[0]
        healthy_count = healthy_indices.shape[0]

        hd_ratio = healthy_count / disease_count
        print("hd_ratio, healthy_count, disease_count, self._labels.shape : ", hd_ratio, healthy_count, disease_count, self._labels.shape)
        # assert False, (np.unique(self._labels))
        # assert False, (hd_ratio, healthy_count, disease_count, self._labels.shape)

        if hd_ratio >= 1:
            disease_repeater = np.repeat(disease_indices, np.ceil(hd_ratio))
            disease_repeater = np.random.permutation(disease_repeater)[:healthy_count]
            up_idx = np.concatenate( (healthy_indices, disease_repeater) )
        else:
            healthy_repeater = np.repeat(healthy_indices, np.ceil(1/hd_ratio))
            healthy_repeater = np.random.permutation(healthy_repeater)[:disease_count]
            up_idx = np.concatenate( (healthy_repeater, disease_indices) )

        # up_idx = np.concatenate( (np.arange(len(self._labels)), np.repeat(pos_idx, hd_ratio)))
        # up_idx = up_idx[:2*healthy_count]
        self._image_paths = self._image_paths[up_idx]
        self._labels = self._labels[up_idx]
        if self._hosp is not None:
            self._hosp = self._hosp[up_idx]
            
    def downsample(self):
        assert self._labels.shape[1] == 1
        # ratio = (len(self._labels) - self._labels.sum(axis=0)
        #          ) // self._labels.sum(axis=0) - 1
        # ratio = ratio[self._idx][0]
        # assert False, (self._idx, self._labels.shape, ratio)
        indices = np.arange(len(self._labels))
        disease_indices = indices[(self._labels[:, self._idx] > 0).ravel()]
        healthy_indices = indices[(self._labels[:, self._idx] < 1).ravel()]
        disease_count = disease_indices.shape[0]
        healthy_count = healthy_indices.shape[0]

        hd_ratio = healthy_count / disease_count
        print("hd_ratio, healthy_count, disease_count, self._labels.shape : ", hd_ratio, healthy_count, disease_count, self._labels.shape)
        # assert False, (np.unique(self._labels))
        # assert False, (hd_ratio, healthy_count, disease_count, self._labels.shape)

        if hd_ratio >= 1:
            # healthy is larger; cut it down to diseased
            health_index_subset = np.random.permutation(healthy_indices)[:disease_count]
            down_idx = np.concatenate( (health_index_subset, disease_indices) )
        else:
            # healthy is smaller; cut down the diseased dataset
            disease_index_subset = np.random.permutation(disease_indices)[:healthy_count]
            down_idx = np.concatenate( (healthy_indices, disease_index_subset) )

        # down_idx = np.concatenate( (np.arange(len(self._labels)), np.repeat(pos_idx, hd_ratio)))
        # down_idx = down_idx[:2*healthy_count]
        self._image_paths = self._image_paths[down_idx]
        self._labels = self._labels[down_idx]
        if self._hosp is not None:
            self._hosp = self._hosp[down_idx]
            
class JointDataset(MultipleDomainDataset):
    ENVIRONMENTS = ['mimic-cxr', 'chexpert']
    N_STEPS = 100000  # Default, subclasses may override
    CHECKPOINT_FREQ = 5000  # Default, subclasses may override
    N_WORKERS = 8

    def __init__(self, mode, hparams, _datasetTWO="mimic"):
        # ONE OF THE DATSETS IS ALWAYS CHEXPERT, SECOND DATASET IS CHANGED
        super().__init__()
        assert "label_balance_method" in hparams.keys(), hparams.keys()
        print("========== CREATING A NEW JOINT DATASET ==========")
        if hparams['change_disease'] is not None:
            # assert False
            global diseases
            diseases = [hparams['change_disease']]
            print("THE NEW DISEASE SET IS : ", diseases)
        print('only using {} and CHEXPERT'.format(_datasetTWO))
        print('only using {} and CHEXPERT'.format(_datasetTWO))
        print('only using {} and CHEXPERT'.format(_datasetTWO))
        print('only using {} and CHEXPERT'.format(_datasetTWO))
        #     label_paths={
        #         'mimic':"/scratch/wz727/chestXR/data/mimic-cxr/train_sub.csv",
        #         'chexpert":"/CheXpert-v1.0/train.csv"
        #     }
        # paths = ['/beegfs/wz727/mimic-cxr',
        #          '/scratch/wz727/chest_XR/chest_XR/data/CheXpert',
        #          '/scratch/wz727/chest_XR/chest_XR/data/chestxray8',
        #          '/scratch/lhz209/padchest']
        # paths = ['/scratch/wz727/chestXR/data/mimic-cxr', '', '/chestxray8', '/padchest']
        print('CALLING {}'.format(_datasetTWO))
        DATASETS = {
            'mimic' : (MimicCXRDataset, "/scratch/wz727/chestXR/data/mimic-cxr/train_sub.csv"),
            'padchest' : (PadChestDataset, "/padchest/")
        }
        _DATASET_CALL, _CALL_STRING = DATASETS[_datasetTWO]
        
        datasetTWO = _DATASET_CALL(_CALL_STRING,
                                    mode=mode, upsample=False,
                                    subset=hparams['subset'],
                                    input_shape=hparams['input_shape'],
                                    hparams=hparams)
            
        #     MimicCXRDataset("/scratch/wz727/chestXR/data/mimic-cxr/train_sub.csv",
        #                             )
        print('CALLING CHEXPERT')
        chexpert = CheXpertDataset('/CheXpert-v1.0/train_sub.csv', mode=mode,
                                   upsample=False,
                                   subset=hparams['subset'],
                                   input_shape=hparams['input_shape'])

        # TODO: setup ability to call other datasets

        chexpert_x = chexpert._image_paths
        chexpert_y = chexpert._labels.ravel().astype(np.int)

        datasetTWO_x = datasetTWO._image_paths
        datasetTWO_y = datasetTWO._labels.ravel().astype(np.int)

        print("=====================================================================")
        print("=====================================================================")
        print("")
        # get counts
        # print('GETTING COUNTS')
        # datasetTWO_count = np.sum(datasetTWO_y)
        # chexpert_count = np.sum(chexpert_y)
        print('Datasets original Y means {}, {}'.format( datasetTWO_y.mean(), chexpert_y.mean()))
        # datasetTWO_healthy_count = datasetTWO_y.shape[0] - datasetTWO_count
        # chexpert_healthy_count = chexpert_y.shape[0] - chexpert_count

        # find disease case indices
        datasetTWO_disease_indices = np.arange(datasetTWO_y.shape[0])[datasetTWO_y > 0]
        chexpert_disease_indices = np.arange(
            chexpert_y.shape[0])[chexpert_y > 0]

        # find healthy case indices
        datasetTWO_healthy_indices = np.arange(datasetTWO_y.shape[0])[datasetTWO_y == 0]
        chexpert_healthy_indices = np.arange(
            chexpert_y.shape[0])[chexpert_y == 0]

        def _sample_with_prevalence_diffs(indices_a, indices_b, hold=None, rho=0.9):
            len_a = len(indices_a)
            len_b = len(indices_b)
            
            # we will let len_a / rho be total count and derive 
            if len_b >= int(len_a/rho) - len_a:
                # if we can get sufficient samples from b, we're good
                total_count = min(int(len_a/rho), KEEP_COUNT_MAX)
                indices_a_keep_count = int(np.floor(rho*total_count))
                indices_b_keep_count = total_count - indices_a_keep_count

            else:
                assert len_a >= int(len_b/(1-rho)) - len_b
            # if don't, then we star with  get sufficient samples from b, we're good
                total_count = min(int(len_b/(1-rho)), KEEP_COUNT_MAX)
                indices_b_keep_count = int(np.floor((1-rho)*total_count))
                indices_a_keep_count = total_count - indices_b_keep_count

            print(" ==== ", rho, indices_a_keep_count, indices_b_keep_count)

            return np.random.choice(indices_a, indices_a_keep_count), np.random.choice(indices_b, indices_b_keep_count)

        # Gettting rho from hparams; the mode decides this.
        rho = hparams['rho'] if mode in ['train', 'dry'] else 1 - hparams['rho_test'] # convention : rho_test=0.9 is equally strong as rho=0.9 but opposite
        # rho = hparams['rho']
        # rho_test = hparams['rho_test']

        # CONSTRUCT THE TRAIN DISTRIBUTION
        # print('SET TRAIN PREVALENCE')
        datasetTWO_healthy_keep, chexpert_healthy_keep = _sample_with_prevalence_diffs(
            datasetTWO_healthy_indices, chexpert_healthy_indices, rho=1 - rho)

        datasetTWO_disease_keep, chexpert_disease_keep = _sample_with_prevalence_diffs(
            datasetTWO_disease_indices, chexpert_disease_indices, rho=rho)

        datasetTWO_train = np.concatenate(
            (datasetTWO_healthy_keep, datasetTWO_disease_keep))
        chexpert_train = np.concatenate(
            (chexpert_healthy_keep, chexpert_disease_keep))

        chexpert_cfgstr = domainbed_path + '/configs/chexpert_config.json'
        with open(chexpert_cfgstr) as f:
            chexpert_cfg = edict(json.load(f))

        datasetTWO_cfgstr = domainbed_path + '/configs/chexpert_config.json'
        with open(datasetTWO_cfgstr) as f:
            datasetTWO_cfg = edict(json.load(f))

        biased_cxr_dataset = ChestDataset()
        biased_cxr_dataset._image_paths = np.concatenate(
            (datasetTWO_x[datasetTWO_train], chexpert_x[chexpert_train]))
        biased_cxr_dataset._labels = np.concatenate(
            (datasetTWO_y[datasetTWO_train], chexpert_y[chexpert_train])).reshape(-1, 1)
        biased_cxr_dataset._idx = np.array([diseases.index(d) for d in diseases])
        if hparams['hosp']:
            biased_cxr_dataset._hosp = np.concatenate(
                (datasetTWO_y[datasetTWO_train]*0  + 1 , chexpert_y[chexpert_train]*0)).reshape(-1, 1)
            # this ensures all datasetTWO examples get hospital 1 and all chexpert ones get hospital 0
        else:
            biased_cxr_dataset._hosp = None
        print('JOINT {} Datasets before upsampling with Y means {}'.format(mode, biased_cxr_dataset._labels.mean()))
        if hparams['label_balance_method'] == "upsample":
            biased_cxr_dataset.upsample()
        elif hparams['label_balance_method'] == "downsample":
            biased_cxr_dataset.downsample()
        else:
            assert False, "balance method unknown!"
        biased_cxr_dataset.cfg = datasetTWO_cfg
        biased_cxr_dataset._num_image = len(biased_cxr_dataset._labels)
        biased_cxr_dataset._mode = mode
        biased_cxr_dataset.input_shape=hparams['input_shape']
        biased_cxr_dataset.aug_transform = hparams.get('aug_transform', False)

        self.dataset=biased_cxr_dataset
        self.input_shape = hparams['input_shape'] # (1, 256, 256,)
        self.num_classes = 1

        y_long = torch.from_numpy(biased_cxr_dataset._labels).long().view(-1)
        z_long = torch.from_numpy(biased_cxr_dataset._hosp).long().view(-1)

        print('JOINT dataset generated with Y mean {}'.format(biased_cxr_dataset._labels.mean()))
        print('JOINT dataset generated with bias   {}'.format((y_long == z_long).float().mean()))

        # GET SAMPLES NOT IN THE TRAIN DISTRIBUTION
        # datasetTWO_healthy_remain = np.setdiff1d(
        #     datasetTWO_healthy_indices, datasetTWO_healthy_train)
        # chexpert_healthy_remain = np.setdiff1d(
        #     chexpert_healthy_indices, chexpert_healthy_train)
        # datasetTWO_disease_remain = np.setdiff1d(
        #     datasetTWO_disease_indices, datasetTWO_disease_train)
        # chexpert_disease_remain = np.setdiff1d(
        #     chexpert_disease_indices, chexpert_disease_train)

        # CONSTRUCT THE TEST DISTRIBUTION
        # print('SET TEST PREVALENCE')
        # datasetTWO_healthy_test, chexpert_healthy_test = _sample_with_prevalence_diffs(
        #     datasetTWO_healthy_remain, chexpert_healthy_remain, rho=rho_test)
        # datasetTWO_disease_test, chexpert_disease_test = _sample_with_prevalence_diffs(
        #     datasetTWO_disease_remain, chexpert_disease_remain, rho=1-rho_test)

        # datasetTWO_test = np.concatenate((datasetTWO_healthy_test, datasetTWO_disease_test))
        # chexpert_test = np.concatenate(
        #     (chexpert_healthy_test, chexpert_disease_test))

        # config creation
        # chexpert_cfgstr = domainbed_path + '/configs/chexpert_config.json'
        # with open(chexpert_cfgstr) as f:
        #     chexpert_cfg = edict(json.load(f))

        # datasetTWO_cfgstr = domainbed_path + '/configs/chexpert_config.json'
        # with open(datasetTWO_cfgstr) as f:
        #     datasetTWO_cfg = edict(json.load(f))

        # biased_cxr_dataset = ChestDataset()
        # biased_cxr_dataset._image_paths = np.concatenate(
        #     (datasetTWO_x[datasetTWO_train], chexpert_x[chexpert_train]))
        # biased_cxr_dataset._labels = np.concatenate(
        #     (datasetTWO_y[datasetTWO_train], chexpert_y[chexpert_train])).reshape(-1, 1)
        # biased_cxr_dataset._idx = np.array([diseases.index(d) for d in diseases])
        # if hparams['hosp']:
        #     biased_cxr_dataset._hosp = np.concatenate(
        #         (datasetTWO_y[datasetTWO_train]*0  + 1 , chexpert_y[chexpert_train]*0)).reshape(-1, 1)
        #     # this ensures all datasetTWO examples get hospital 1 and all chexpert ones get hospital 0
        # else:
        #     biased_cxr_dataset._hosp = None
        # print('JOINT {} Datasets before upsampling with Y means {}'.format(mode, biased_cxr_dataset._labels.mean()))
        # if hparams['label_balance_method'] == "upsample":
        #     biased_cxr_dataset.upsample()
        # elif hparams['label_balance_method'] == "downsample":
        #     biased_cxr_dataset.downsample()
        # else:
        #     assert False, "balance method unknown!"
        # biased_cxr_dataset.cfg = datasetTWO_cfg
        # biased_cxr_dataset._num_image = len(biased_cxr_dataset._labels)
        # biased_cxr_dataset._mode = mode
        # biased_cxr_dataset.input_shape=hparams['input_shape']
        # biased_cxr_dataset.aug_transform = hparams.get('aug_transform', False)

        # cxr_test = ChestDataset()
        # cxr_test._image_paths = np.concatenate(
        #     (datasetTWO_x[datasetTWO_test], chexpert_x[chexpert_test]))
        # cxr_test._labels = np.concatenate(
        #     (datasetTWO_y[datasetTWO_test], chexpert_y[chexpert_test])).reshape(-1, 1)
        # cxr_test._idx = np.array([diseases.index(d) for d in diseases])
        # if hparams['hosp']:
        #     cxr_test._hosp = np.concatenate(
        #         (datasetTWO_y[datasetTWO_test]*0  + 1 , chexpert_y[chexpert_test]*0)).reshape(-1, 1)
        #     # this ensures all datasetTWO examples get hospital 1 and all chexpert ones get hospital 0
        # else:
        #     cxr_test._hosp = None
        
        # print("TEST HOSPITAL SPLIT FOR DISEASE", torch.unique(torch.from_numpy(cxr_test._hosp[cxr_test._labels==1]), return_counts=True))
        # print('JOINT TEST Datasets before upsampling with Y means {}'.format( cxr_test._labels.mean()))
        # if hparams['label_balance_method'] == "upsample":
        #     cxr_test.upsample()
        # elif hparams['label_balance_method'] == "downsample":
        #     cxr_test.downsample()
        # else:
        #     assert False, "balance method unknown!"
        # cxr_test.cfg = datasetTWO_cfg
        # cxr_test._num_image = len(cxr_test._labels)
        # cxr_test._mode = mode
        # cxr_test.input_shape=hparams['input_shape']
        # cxr_test.aug_transform = hparams.get('aug_transform', False)

        # self.datasets = [biased_cxr_dataset, cxr_test]
        # print('JOINT Datasets generated with Y means {}, {}'.format( biased_cxr_dataset._labels.mean(), cxr_test._labels.mean()))
        # print('JOINT Datasets generated with Y shapes {}, {}'.format( biased_cxr_dataset._labels.shape, cxr_test._labels.shape))

        # y_mode = biased_cxr_dataset._labels.astype(long())
        # h_mode = biased_cxr_dataset._hosp.long()

        # print(" MODE {}, CORR = {}".format(mode,(y_mode == h_mode).float().mean()))
        # print(" MODE {}, CORR = {}".format(mode,(y_mode == h_mode).float().mean()))
        # print(" MODE {}, CORR = {}".format(mode,(y_mode == h_mode).float().mean()))
        # print(" MODE {}, CORR = {}".format(mode,(y_mode == h_mode).float().mean()))

        # self.input_shape = hparams['input_shape'] # (1, 256, 256,)
        # self.num_classes = 1


class CheXpertDataset(ChestDataset):
    def __init__(self, label_path, cfg=domainbed_path + '/configs/chexpert_config.json', mode='train', upsample=True, subset=True, input_shape=(1, 64, 64), aug_transform=False, hparams=None):
        label_path = None # this is over written later.
        assert mode in ['train', 'val', 'dry'], 'only train, val, dry allows as mode'
        print("========== CREATING A NEW CHEXPERT DATASET ==========")
        self._hosp = None
        self.input_shape = input_shape
        self.aug_transform = aug_transform
        if hparams is not None and hparams['change_disease'] is not None:
            global diseases
            diseases = [hparams['change_disease']]
            print("THE NEW DISEASE SET IS : ", diseases)
        def get_labels(labels):
            all_labels = []
            for _, row in labels.iterrows():
                all_labels.append([row[d] in [1, -1] for d in diseases])
            return all_labels

        def get_image_paths(labels):
            self._data_path = ''
            all_paths = []
            for _, row in labels.iterrows():
                all_paths.append(self._data_path + '/' + row['Path'])
            return all_paths

        with open(cfg) as f:
            self.cfg = edict(json.load(f))
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        labels_path = csv_path + 'chexpert_{}.csv'.format('train' if mode=='dry' else mode)
        df = pd.read_csv(labels_path)
        # subsetting the data
        # TODO: validation at some point should not be subsetted
        if subset:
            uncertain_diseases = [
                d for d in diseases if d in ['Atelectasis', 'Edema']]
            if uncertain_diseases:
                mask = (df[diseases + ['No Finding']] ==
                        1).any(1) | (df[uncertain_diseases] == -1).any(1)
            else:
                mask = (df[diseases + ['No Finding']] == 1).any(1)
            labels = df[mask]
        else:
            labels = df
        labels.fillna(0, inplace=True)
        self._labels = get_labels(labels)
        self._image_paths = get_image_paths(labels)
        self._idx = np.array([diseases.index(d) for d in diseases])
        self._image_paths = np.array(self._image_paths)
        self._labels = np.array(self._labels)

        # KEEPING ONLY FRONTAL
        # print(self._image_paths[:1000])
        if mode=='dry':
            self._labels = self._labels[:100]
            self._image_paths = self._image_paths[:100]

        if upsample:
            ratio = (len(self._labels) - self._labels.sum(axis=0)
                     ) // self._labels.sum(axis=0) - 1
            ratio = ratio[self._idx][0]
            # print('IDX THING HAPPENING')
            pos_idx = np.where(self._labels[:, self._idx] == 1)[0]
            if ratio >= 1:
                up_idx = np.concatenate(
                    (np.arange(len(self._labels)), np.repeat(pos_idx, ratio)))
                self._image_paths = self._image_paths[up_idx]
                self._labels = self._labels[up_idx]
        self._labels = self._labels[:, self._idx]
        self._num_image = len(self._image_paths)
        print('CONSTRUCTED CHEXPERT DATA WITH LABEL MEAN {}/ SHAPE {}'.format( self._labels.ravel().mean(), self._labels.shape))


class MimicCXRDataset(ChestDataset):
    def __init__(self, label_path, cfg=domainbed_path+'/configs/mimic_config.json', mode='train', upsample=True, subset=True, input_shape=[1,64,64], aug_transform=False, hparams=None):
        label_path = None # this is over written later.
        assert mode in ['train', 'val', 'dry'], 'only train, val, dry allows as mode'
        print("========== CREATING A NEW MIMIC DATASET ==========")
        self._hosp = None
        self.input_shape = input_shape
        self.aug_transform = aug_transform
        if hparams is not None and hparams['change_disease'] is not None:
            global diseases
            diseases = [hparams['change_disease']]
            print("THE NEW DISEASE SET IS : ", diseases)
        def get_labels(labels):
            all_labels = []
            for _, row in labels.iterrows():
                all_labels.append([row[d] in [1, -1] for d in diseases])
            return all_labels

        def get_image_paths(labels):
            all_paths = []
            for _, row in labels.iterrows():
                if int(str(row['subject_id'])[:2]) < 15:
                    data_path = '/mimic-cxr_1'
                else:
                    data_path = '/mimic-cxr_2'
                all_paths.append(
                    data_path + '/p' + str(row['subject_id'])[:2] + '/p' + str(row['subject_id']) +
                    '/s' + str(row['study_id']) + '/' +
                    str(row['dicom_id']) + '.jpg'
                )
            return all_paths

        with open(cfg) as f:
            self.cfg = edict(json.load(f))
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        label_path = csv_path + 'mimic_original_{}.csv'.format('train' if mode=='dry' else mode)
        df = pd.read_csv(label_path)

        # subsetting the data
        # TODO: validation at some point should not be subsetted
        if subset:
            uncertain_diseases = [
                d for d in diseases if d in ['Atelectasis', 'Edema']]
            if uncertain_diseases:
                mask = (df[diseases + ['No Finding']] ==
                        1).any(1) | (df[uncertain_diseases] == -1).any(1)
            else:
                mask = (df[diseases + ['No Finding']] == 1).any(1)
            labels = df[mask]
        else:
            labels = df
        labels = labels[pd.notnull(labels['dicom_id'])]
        # print(labels.shape)
        # labels = labels[labels['ViewPosition'].isin(["PA", "AP"])]
        # assert False, (labels.shape,labels['ViewPosition'].unique())
        # assert False, labels.columns
        labels.fillna(0, inplace=True)
        self._labels = get_labels(labels)
        self._image_paths = get_image_paths(labels)
        self._idx = np.array([diseases.index(d) for d in diseases])
        self._image_paths = np.array(self._image_paths)
        self._labels = np.array(self._labels)
        # self._hosp = self._labels*0 + 1

        if hparams is not None and hparams.get('create_dataset', False):
            if mode in ["train", "dry"]:
                # with open("/scratch/apm470/nuisance-orthogonal-prediction/code/nrd-xray/mimic_keep_train.npz", 'rb') as f:
                #     remove_paths = np.load(f)

                with open("./mimic_keep_bigcut_train.npz", 'rb') as f:
                    mimic_keep_vals = np.load(f)

            if mode == "val":
                # with open("/scratch/apm470/nuisance-orthogonal-prediction/code/nrd-xray/mimic_keep_val.npz", 'rb') as f:
                #     remove_paths = np.load(f)

                with open("./mimic_keep_bigcut_val.npz", 'rb') as f:
                    mimic_keep_vals = np.load(f)

                
            print("FILTERING STEP HAPPENING")
            print("FILTERING STEP HAPPENING")
            print(" ---- before filtering shapes", (self._image_paths.shape, self._labels.shape))
            # mimic_keep_vals = np.setdiff1d(self._image_paths, remove_paths, assume_unique=True) # get paths that we want to keep
            _, mimic_keep_indices, _ = np.intersect1d(self._image_paths, mimic_keep_vals, assume_unique=True, return_indices=True) # get indices of paths we want to keep from the existing paths
            self._image_paths = self._image_paths[mimic_keep_indices]
            self._labels = self._labels[mimic_keep_indices]
            print(" ---- after filtering shapes", (self._image_paths.shape, self._labels.shape))
            print("FILTERING STEP DONE")
            print("FILTERING STEP DONE")

        if mode=='dry':
            self._labels = self._labels[:100]
            self._image_paths = self._image_paths[:100]

        if upsample:
            ratio = (len(self._labels) - self._labels.sum(axis=0)
                     ) // self._labels.sum(axis=0) - 1
            ratio = ratio[self._idx][0]
            pos_idx = np.where(self._labels[:, self._idx] == 1)[0]
            if ratio >= 1:
                up_idx = np.concatenate(
                    (np.arange(len(self._labels)), np.repeat(pos_idx, ratio)))
                self._image_paths = self._image_paths[up_idx]
                self._labels = self._labels[up_idx]
        self._labels = self._labels[:, self._idx]
        self._num_image = len(self._image_paths)
        print('CONSTRUCTED MIMIC DATA WITH LABEL MEAN {}/ SHAPE {}'.format( self._labels.ravel().mean(), self._labels.shape))


class ChestXR8Dataset(ChestDataset):
    def __init__(self, label_path, cfg='configs/chestxray8_config.json', mode='train', upsample=True, subset=True):
        label_path = None # this is specified later
        def get_labels(label_strs):
            all_labels = []
            for label in label_strs:
                labels_split = label.split('|')
                label_final = [d in labels_split for d in diseases]
                all_labels.append(label_final)
            return all_labels

        self._data_path = label_path.rsplit('/', 1)[0]
        self._mode = mode
        with open(cfg) as f:
            self.cfg = edict(json.load(f))
        label_path = csv_path + 'chestxray8_{}.csv'.format(mode)
        labels = pd.read_csv(label_path)
        if subset:
            labels = labels[labels['Finding Labels'].str.contains(
                '|'.join(diseases + ['No Finding']))]
        if self._mode == 'train' and upsample:
            # labels_neg = labels[labels['Finding Labels'].str.contains('No Finding')]
            # labels_pos = labels[~labels['Finding Labels'].str.contains('No Finding')]
            # one vs all
            labels_pos = labels[labels['Finding Labels'].str.contains(
                diseases[0])]
            labels_neg = labels[~labels['Finding Labels'].str.contains(
                diseases[0])]
            upweight_ratio = len(labels_neg)//len(labels_pos)
            if upweight_ratio > 0:
                labels_pos = labels_pos.loc[labels_pos.index.repeat(
                    upweight_ratio)]
                labels = pd.concat([labels_neg, labels_pos])
        self._image_paths = [os.path.join(
            self._data_path, 'images', name) for name in labels['Image Index'].values]
        self._labels = get_labels(labels['Finding Labels'].values)
        self._num_image = len(self._image_paths)


class PadChestDataset(ChestDataset):
    def __init__(self, label_path, cfg=domainbed_path+'/configs/padchest_config.json', mode='train', upsample=True, subset=True, input_shape=[1,64,64], aug_transform=False, hparams=None):
        assert mode in ['train', 'val', 'dry'], mode
        self._hosp = None
        self.input_shape = input_shape
        self.aug_transform = aug_transform
        def get_labels(label_strs):
            all_labels = []
            for label in label_strs:
                label_final = [d.lower() in label for d in diseases]
                all_labels.append(label_final)
            return all_labels

        self._data_path = label_path.rsplit('/', 1)[0]
        # assert False, self._data_path
        self._mode = "train" if mode == "dry" else mode
        with open(cfg) as f:
            self.cfg = edict(json.load(f))
        label_path = csv_path + 'padchest_{}.csv'.format(self._mode)
        labels = pd.read_csv(label_path)
        positions = ['AP', 'PA', 'ANTEROPOSTERIOR', 'POSTEROANTERIOR']
        labels = labels[
            pd.notnull(labels['ViewPosition_DICOM']) & labels['ViewPosition_DICOM'].str.match('|'.join(positions))]
        labels = labels[pd.notnull(labels['Labels'])]
        if subset:
            labels = labels[labels['Labels'].str.contains(
                '|'.join([d.lower() for d in diseases] + ['normal']))]

        if upsample:
            # labels_neg = labels[labels['Labels'].str.contains('normal')]
            # labels_pos = labels[~labels['Labels'].str.contains('normal')]
            # one vs all
            labels_pos = labels[labels['Labels'].str.contains(
                diseases[0].lower())]
            labels_neg = labels[~labels['Labels'].str.contains(
                diseases[0].lower())]
            upweight_ratio = len(labels_neg)//len(labels_pos)
            if upweight_ratio > 0:
                labels_pos = labels_pos.loc[labels_pos.index.repeat(
                    upweight_ratio)]
                labels = pd.concat([labels_neg, labels_pos])
        self._image_paths = [os.path.join(
            self._data_path, name) for name in labels['ImageID'].values]
        self._labels = get_labels(labels['Labels'].values)
        if mode=='dry':
            self._labels = self._labels[:100]
            self._image_paths = self._image_paths[:100]

        self._image_paths = np.array(self._image_paths)
        self._labels = np.array(self._labels)
        self._num_image = len(self._image_paths)
        # print(self._image_paths)
        # assert False
        print('CONSTRUCTED PADCHEST DATA WITH LABEL MEAN {}/ SHAPE {}'.format( self._labels.ravel().mean(), self._labels.shape))


# class chestXR(MultipleDomainDataset):
#     ENVIRONMENTS = ['mimic-cxr', 'chexpert', 'chestxr8', 'padchest']
#     N_STEPS = 100000  # Default, subclasses may override
#     CHECKPOINT_FREQ = 5000  # Default, subclasses may override
#     N_WORKERS = 8

#     def __init__(self, root, test_envs, mode, hparams):
#         super().__init__()
#         # paths = ['/beegfs/wz727/mimic-cxr',
#         #          '/scratch/wz727/chest_XR/chest_XR/data/CheXpert',
#         #          '/scratch/wz727/chest_XR/chest_XR/data/chestxray8',
#         #          '/scratch/lhz209/padchest']
#         paths = ['/scratch/wz727/chestXR/data/mimic-cxr',
#                  '', '/chestxray8', '/padchest']
#         self.datasets = []
#         for i, environment in enumerate(chestXR.ENVIRONMENTS):
#             print(environment)
#             path = os.path.join(root, environment)
#             if environment == 'mimic-cxr':
#                 env_dataset = MimicCXRDataset(
#                     paths[i] + '/train_sub.csv', mode=mode, upsample=hparams['upsample'], subset=hparams['subset'])
#             elif environment == 'chexpert':
#                 env_dataset = CheXpertDataset(
#                     paths[i] + '/CheXpert-v1.0/train_sub.csv', mode=mode, upsample=hparams['upsample'], subset=hparams['subset'])
#             elif environment == 'chestxr8':
#                 env_dataset = ChestXR8Dataset(
#                     paths[i] + '/Data_Entry_2017_v2020.csv', mode=mode, upsample=hparams['upsample'], subset=hparams['subset'])
#             elif environment == 'padchest':
#                 env_dataset = PadChestDataset(
#                     paths[i] + '/padchest_labels.csv', mode=mode, upsample=hparams['upsample'], subset=hparams['subset'])
#             else:
#                 raise Exception('Unknown environments')
#             if mode != 'train':
#                 env_dataset.cfg.use_transforms_type = 'None'

#             self.datasets.append(env_dataset)

#         self.input_shape = (3, 512, 512,)
#         self.num_classes = 1


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


# class Debug(MultipleDomainDataset):
#     def __init__(self, root, test_envs, hparams):
#         super().__init__()
#         self.input_shape = self.INPUT_SHAPE
#         self.num_classes = 2
#         self.datasets = []
#         for _ in [0, 1, 2]:
#             self.datasets.append(
#                 TensorDataset(
#                     torch.randn(16, *self.INPUT_SHAPE),
#                     torch.randint(0, self.num_classes, (16,))
#                 )
#             )


# class Debug28(Debug):
#     INPUT_SHAPE = (3, 28, 28)
#     ENVIRONMENTS = ['0', '1', '2']