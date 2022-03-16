import os
from re import A
import sys
import matplotlib.pyplot as plt
import utils

import numpy as np
import PIL
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split, Dataset

from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.utils import save_image

import sys
from datasets import CheXpertDataset, MimicCXRDataset, JointDataset, PadChestDataset
import utils
from utils import get_random_subset, split_20percent_array, get_color

from os.path import dirname
filepath = os.path.abspath(__file__)
root_dir = dirname(filepath)
# erm_dir = dirname(filepath) + "/erm-on-generated/"

CAN_LOAD_WATERBIRDS=True
print(root_dir + "/group_dro_data/")
if os.path.isdir(root_dir + "/group_dro_data/"):
    sys.path.append(root_dir + "/group_dro_data/")
    from cub_dataset import CUBDataset
    from dro_dataset import DRODataset
    WATERBIRDS_ROOT_DIR=root_dir + "/cub/"
    assert os.path.isdir(WATERBIRDS_ROOT_DIR)
else:
    CAN_LOAD_WATERBIRDS=False

# class XYZ_DatasetWithIndices(TensorDataset):
#     def __init__(self, *tensors):
#         super().__init__(*tensors)
#         self.weights = None
#         self.y = tensors[1]

#     def __getitem__(self, index):
#         if self.weights is not None:
#             return tuple(tensor[index] for tensor in self.tensors) + tuple([self.weights[index], index])
#         else:
#             return tuple(tensor[index] for tensor in self.tensors) + tuple([index])

class XYZ_DatasetWithIndices(Dataset):
    def __init__(self, x, y, z=None):
        self.weights = None
        self.x = x
        self.y = y
        self.z = z
        self.weights=None
    
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        if self.weights is not None:
            return (self.x[index], self.y[index], self.z[index], self.weights[index], torch.tensor(index))
        else:
            return (self.x[index], self.y[index], self.z[index], torch.tensor(index))

def create_loader(_dataset, LOADER_PARAMS, weights=None, shuffle=False, strategy="plain"):
    # when strategy=="weight" or "plain", a normal dataloader is created
    if strategy == "upsample":
        assert weights is not None
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=weights.shape[0],
            replacement=True
        )

    elif strategy == "downsample":
        raise NotImplemented
    else:
        assert strategy in ["weight", "plain"], "unknown strategy of nuisance randomization."
        sampler = None

    return DataLoader(
        _dataset,
        batch_size=LOADER_PARAMS['BATCH_SIZE'],
        num_workers=LOADER_PARAMS['workers'],
        pin_memory=LOADER_PARAMS['pin_memory'],
        drop_last=False,
        shuffle=shuffle and strategy != 'upsample',
        sampler=sampler
    )

def generate_synthetic_data_discz(a, m):
    assert a > 0
    assert a < 1

    y = torch.bernoulli(0.5*torch.ones(size=(m,))).long()

    z_g_y_probs = a*torch.ones(y.shape)
    z_g_y_probs[y==1] = a
    z_g_y_probs[y==0] = 1 - a
    z = torch.bernoulli(z_g_y_probs)

    eps_1 = torch.randn((m, ))
    eps_2 = torch.randn((m, ))

    assert y.shape == eps_2.shape

    x = torch.cat([(y - z + eps_1).view(-1, 1,),
                  (y + z + eps_2).view(-1, 1,)], dim=1)

    xz = torch.cat([x, z.view(-1, 1)], dim=1)

    return XYZ_DatasetWithIndices(x=xz, y=y.long(), z=z.long())


def generate_synthetic_data(a, m=5000):
    y = torch.randint(low=0, high=2, size=(m, )).long()

    # z_g_y_probs = rho*torch.ones(y.shape)
    # z_g_y_probs[y==1] = rho
    # z_g_y_probs[y==0] = 1 - rho
    # z = torch.bernoulli(z_g_y_probs)
    z = torch.randn((m, )) + a*(2*y - 1)

    y_polar = 2*y - 1
    # z_polar = 2*z - 1

    eps_1 = torch.randn((m, ))
    eps_2 = torch.randn((m, ))

    # assert y.shape == z_polar.shape
    assert y.shape == eps_2.shape

    # x = torch.cat([(radius*cos_angle).view(-1,1), (radius*sin_angle).view(-1,1)], dim=1)
    # x= torch.cat([(y - 0.5*z + 0.7*eps_1).view(-1,1,), (y - z + 0.7*eps_2).view(-1,1,)], dim=1)
    x = torch.cat([(y - z + 3*eps_1).view(-1, 1,),
                  (y + z + 0.1*eps_2).view(-1, 1,)], dim=1)

    xz = torch.cat([x, z.view(-1, 1)], dim=1)

    return XYZ_DatasetWithIndices(x=xz, y=y.long(), z=z)


def subsample_to_balance_subset(_subset):
    print("-")
    _subset_indices = _subset.indices
    _subset_g = _subset.dataset.group_array[_subset_indices]
    print("UNIQUE BEFORE", np.unique(_subset_g, return_counts=True))
    print("Y_MEAN BEFORE", np.mean(_subset.dataset.y_array[_subset_indices]))

    _y0_indices = _subset_indices[_subset_g < 2]
    _y1_indices = _subset_indices[_subset_g >= 2]

    _y0_size = len(_y0_indices)
    _y1_size = len(_y1_indices)

    assert _y0_size > _y1_size, (_y0_size, _y1_size)

    perm = torch.randperm(_y1_size)
    _y0_indices = _y0_indices[perm[:_y0_size]]

    _subset_indices = np.concatenate((_y1_indices, _y0_indices), axis=0)
    _subset_g = _subset.dataset.group_array[_subset_indices]

    print("UNIQUE AFTER", np.unique(_subset_g, return_counts=True))
    # assert False, [item.shape for item in [_y0_indices, _y1_indices, _subset_indices]]
    print("Y_MEAN AFTER", np.mean(_subset.dataset.y_array[_subset_indices]))

    print("-")

    return Subset(_subset.dataset, _subset_indices)

def _load_train_data_from_filename(args):
    # THIS FUNCTION SHOULD CHANGE BASED ON HOW YOUR DATA IS GENERATED; the dataset is of the class XYZ_DatasetWithIndices with x, y, z where y and z are assumed to be 1-dim
    # when train_filename is given, coeff should specified and exact should be given; otherwise the usecase is out of scope
    assert args.exact and args.coeff > 0
    loaded_data = torch.load(args.train_filename, map_location="cpu")
    if not isinstance(loaded_data, XYZ_DatasetWithIndices):
        assert args.dataset == "cmnist"
        return loaded_data['train']
    else:
        if args.dataset == "joint":
            loaded_data.y = loaded_data.y[:,1] # the y here was created as a one-hot with 1 in the second index for label 1.
        else:
            raise NotImplemented("MAKE SURE THIS FUNCTION PROPERLY LOADS YOUR GENERATED DATA FOR NEW DATASETS.")
        return loaded_data

def construct_dataloaders(args, DATA_HPARAMS, z_loaders=False, load=True, return_datasets_only=False):
    assert not args.eval, "THIS SHOULD BE ENABLED ONLY WHEN DOING NEGATIVE NUISANCE EXPERIMENTS"
    eval_dataset = None
    assert 'rho_test' in DATA_HPARAMS.keys(), DATA_HPARAMS.keys()
    assert abs(args.rho - DATA_HPARAMS['rho']) < 1e-6, (args.rho,
                                                        DATA_HPARAMS['rho'], "rho_test mismatch; PLEASE CALL THE DATASET DIRECTLY")
    assert abs(args.rho_test - DATA_HPARAMS['rho_test']) < 1e-6, (args.rho_test,
                                                                  DATA_HPARAMS['rho_test'], "rho_test mismatch; PLEASE CALL THE DATASET DIRECTLY")
    saved_filename =  dirname(filepath) + "/SAVED_DATA/{}.pt".format(utils.process_save_name(args))

    if os.path.isfile(saved_filename) and load:
        print("FOUND SAVED DATA; USING {}, LOADING!".format(saved_filename))
        datasets = torch.load(saved_filename)

        if args.train_filename is not None:
            print(f"LOADING FROM {args.train_filename}")
            datasets['train'] = _load_train_data_from_filename(args)

            # rewrite the train/val split; this is deterministic. we assume train is shuffled when generating.
            LEN = len(datasets['train'])
            indices = torch.arange(LEN).long()
            m_train = int(0.8*LEN)
            train_indices = indices[:m_train]
            val_indices = indices[m_train:]
            datasets['split'] = {
                'train' : train_indices,
                'val' : val_indices
            }
        for name in ['train', 'test']:
            print(name, datasets[name].x.shape)
        return datasets
    else:
        print("NO SAVED DATA OF NAME {}".format(saved_filename))
        assert not load or args.dataset in ["cmnist", "synthetic"], "always save the dataset if not colored MNIST or synthetic, which are created as tensordatasets"
        
    if args.dataset in utils.waterbirds_list:
        # if not os.path.isdir("/misc/vlgscratch4/RanganathGroup/aahlad/group_DRO/data"):
        #     assert False, "DONT USE GROUP_DRO from here. IT IS NOT MODIFIED CORRECTLY."
        assert CAN_LOAD_WATERBIRDS
        if load:
            assert not args.true_waterbirds
        full_dataset = CUBDataset(
            root_dir=WATERBIRDS_ROOT_DIR,
            target_name="waterbird_complete95",
            confounder_names=["forest2water2"],
            model_type="resnet50",
            augment_data=False,
            data_hparams=DATA_HPARAMS)

        waterbirds_data = DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)

        # TRAIN-TEST SPLIT FIRST
        # making the right sizes
        print("GROUPS = ", np.unique(full_dataset.group_array, return_counts=True))
        saved_split_dict = torch.load("SAVED_DATA/watebirds_split.pt")
        train_group_indices = saved_split_dict['train']
        test_group_indices = saved_split_dict['test']

        MAX_SIZES = [len(train_group_indices[idx]) + len(test_group_indices[idx]) for idx in range(4)]  # p(Y = 0) = 0.75
        print("MAX SIZES ", MAX_SIZES)

        BIAS_TEST = DATA_HPARAMS['rho_test']
        BIAS_TRAIN = DATA_HPARAMS['rho']

        test_total = 200

        K_TEST_BIG = int(test_total*BIAS_TEST)
        K_TEST = test_total - K_TEST_BIG

        test_group_sizes = [K_TEST, K_TEST_BIG, K_TEST_BIG, K_TEST]

        K_TRAIN_INIT = min(MAX_SIZES[2] - K_TEST_BIG, len(train_group_indices[2]))
        K_TRAIN_BIG = min(len(train_group_indices[3]),
            MAX_SIZES[3] - K_TEST, int(K_TRAIN_INIT*(BIAS_TRAIN)/(1 - BIAS_TRAIN)))
        # THIS ENSURES THE RATIO IS CORRECT
        K_TRAIN = int(K_TRAIN_BIG * (1 - BIAS_TRAIN)/(BIAS_TRAIN))
        assert K_TRAIN_INIT >= K_TRAIN, (K_TRAIN_INIT, K_TRAIN)

        train_group_sizes = [K_TRAIN_BIG,
                                K_TRAIN, K_TRAIN, K_TRAIN_BIG]

        for idx in range(4):
            assert train_group_sizes[idx] + \
                test_group_sizes[idx] <= MAX_SIZES[idx]

        print(train_group_sizes, test_group_sizes)

        # train_group_indices, test_group_indices = utils.get_index_subset( train_group_sizes, test_group_sizes, waterbirds_data)

        print("LOADED SPLITs.")

        def filter_from_arrays(list_of_arrays, list_of_sizes):
            assert len(list_of_arrays) == len(list_of_sizes)
            new_list_of_arrays = []
            for ind, array in enumerate(list_of_arrays):
                assert len(array) >= list_of_sizes[ind], (len(
                    array), list_of_sizes[ind])
                new_list_of_arrays.append(array[:list_of_sizes[ind]])
            return new_list_of_arrays

        train_group_indices = filter_from_arrays(
            train_group_indices, train_group_sizes)
        test_group_indices = filter_from_arrays(
            test_group_indices, test_group_sizes)
        
        if not args.eval:
            train_group_indices = torch.cat(train_group_indices, dim=0)
            test_group_indices = torch.cat(test_group_indices, dim=0)

            train_dataset = Subset(waterbirds_data, train_group_indices)
            test_dataset = Subset(waterbirds_data, test_group_indices)
        else:
            print("TOTAL : ", np.array(train_group_sizes) +
                    np.array(test_group_sizes))

            test_group_indices_lists = []
            eval_group_indices_lists = []

            for array in test_group_indices:
                print("SPLITTING TEST AND EVAL")
                _test, _eval = split_20percent_array(array)
                test_group_indices_lists.append(_test)
                eval_group_indices_lists.append(_eval)

                # print(len(_test), len(_eval))

            for name, list_of_lists in [
                ('train', train_group_indices),
                ('test', test_group_indices_lists),
                ('eval', eval_group_indices_lists)
            ]:
                print(f" {name:<6} ", ["{:5d}".format(
                    len(_list)) for _list in list_of_lists])

            # test_group_indices = torch.cat(test_group_indices, dim=0)
            train_group_indices = torch.cat(train_group_indices, dim=0)
            test_group_indices = torch.cat(test_group_indices_lists, dim=0)
            eval_group_indices = torch.cat(eval_group_indices_lists, dim=0)

            lists_to_check = [
                train_group_indices,
                test_group_indices,
                eval_group_indices
            ]

            # just ensuring not common indices
            for idx, tensor_ in enumerate(lists_to_check):
                for _idx, _tensor_ in enumerate(lists_to_check):
                    if idx != _idx:
                        assert len(np.intersect1d(
                            tensor_.numpy(), _tensor_.numpy())) == 0

            train_dataset = Subset(waterbirds_data, train_group_indices)
            test_dataset = Subset(waterbirds_data, test_group_indices)
            eval_dataset = Subset(waterbirds_data, eval_group_indices)

    elif args.dataset == 'synthetic':

        train_dataset = generate_synthetic_data(0.5, m=10000)
        test_dataset = generate_synthetic_data(-0.9, m=2000)


        # train_dataset = generate_synthetic_data_discz(0.9, m=10000)
        # test_dataset = generate_synthetic_data_discz(0.1, m=2000)

        print(train_dataset.x.shape)
        x_train = train_dataset.x[:, :2].cpu()
        y_train = train_dataset.y.cpu().long().view(-1)
        x_test = test_dataset.x[:, :2].cpu()
        y_test = test_dataset.y.cpu().long().view(-1)

        pred_train = x_train[:, 0] + x_train[:, 1]
        pred_test = x_test[:, 0] + x_test[:, 1]

        y_pred = (pred_train > 0.5).long()
        train_acc = (y_train == y_pred).float().mean()
        print("train acc = ", train_acc)
        print("train acc = ", train_acc)
        print("train acc = ", train_acc)
        print("train acc = ", train_acc)

        y_test_pred = (pred_test > 0.5).long()
        test_acc = (y_test == y_test_pred).float().mean()
        print("test acc = ", test_acc)
        print("test acc = ", test_acc)
        print("test acc = ", test_acc)
        print("test acc = ", test_acc)
        # assert False

        if args.debug == 10:
            x_train = train_dataset.x[:, :2].cpu().numpy()
            y_train = train_dataset.y.long().view(-1).cpu().numpy()
            x_test = test_dataset.x[:, :2].cpu().numpy()
            y_test = test_dataset.y.long().view(-1).cpu().numpy()

            plt.figure()
            plt.scatter(x_train[:, 0][y_train == 1], x_train[:, 1]
                        [y_train == 1], label="train y=1", alpha=0.5)
            plt.scatter(x_train[:, 0][y_train == 0], x_train[:, 1]
                        [y_train == 0], label="train y=0", alpha=0.5)
            # plt.scatter(x_test[:, 0][y_test==1], x_test[:, 1][y_test==1], label="test y=1", alpha=0.5)
            # plt.scatter(x_test[:, 0][y_test==0], x_test[:, 1][y_test==0], label="test y=0", alpha=0.5)
            plt.legend()
            plt.savefig("train_synth.png")
            plt.show()

            plt.figure()
            # plt.scatter(x_train[:, 0][y_train==1], x_train[:, 1][y_train==1], label="train y=1", alpha=0.5)
            # plt.scatter(x_train[:, 0][y_train==0], x_train[:, 1][y_train==0], label="train y=0", alpha=0.5)
            plt.scatter(x_test[:, 0][y_test == 1], x_test[:, 1]
                        [y_test == 1], label="test y=1", alpha=0.5)
            plt.scatter(x_test[:, 0][y_test == 0], x_test[:, 1]
                        [y_test == 0], label="test y=0", alpha=0.5)
            plt.legend()
            plt.savefig("test_synth.png")
            plt.show()

        if args.debug > 10:
            assert False

    elif args.dataset == 'joint':
        train_biased_dataset = JointDataset('dry' if args.dry else 'train', DATA_HPARAMS, _datasetTWO=args.datasetTWO)
        test_biased_dataset = JointDataset('dry' if args.dry else 'val', DATA_HPARAMS, _datasetTWO=args.datasetTWO) # val is called for indexing a separate dataset from train. in this codebase, the train is split further into train and val

        train_dataset = train_biased_dataset.dataset
        # THIS IS THE OPPOSITE IMBALANCE FROM THE TRAINING SET
        test_dataset = test_biased_dataset.dataset # called with val
        if args.eval:
            M_TEST = min(1250, len(test_dataset))
            test_dataset, eval_dataset = get_random_subset(
                test_dataset, len(test_dataset), int(0.8*M_TEST), int(0.2*M_TEST))

    elif args.dataset in ['chexpert', 'mimic', 'padchest']:
        _DATASETS = {
            'mimic': (MimicCXRDataset, None),
            'padchest': (PadChestDataset, "/padchest/"),
            'chexpert': (CheXpertDataset, None)
        }
        _DATASET_CALL, _DATASET_STRING = _DATASETS[args.dataset]

        train_dataset = _DATASET_CALL(_DATASET_STRING,
                                        mode='dry' if args.dry else 'train',
                                        input_shape=DATA_HPARAMS['input_shape'],
                                        upsample=True)
        test_dataset = _DATASET_CALL(_DATASET_STRING,
                                    mode='dry' if args.dry else 'val',
                                    input_shape=DATA_HPARAMS['input_shape'],
                                    upsample=True)

    elif args.dataset == "cmnist":
        from datasets_cmnist import ColoredMNIST
        test_envs = [1, 2]
        CMNIST = ColoredMNIST(
            '/scratch/apm470/data/MNIST/', test_envs, DATA_HPARAMS)

        train_env_data_list = []
        ood_env_data_list = []

        for env_i, env in enumerate(CMNIST):
            if env_i not in test_envs:
                train_env_data_list.append(env)
            else:
                ood_env_data_list.append(env)

        assert len(train_env_data_list) == 1, 'WUT'
        train_data = train_env_data_list[0]
        train_x = train_data.tensors[0]  # [:256]
        train_y = train_data.tensors[1][:, 1]  # [:256]

        train_dataset = XYZ_DatasetWithIndices(
                                            train_x,
                                            train_y,
                                            z=utils.get_color(train_x)
                                        )
        # create test data
        test_data = ood_env_data_list[-1]
        test_x = test_data.tensors[0]  # [:256]
        test_y = test_data.tensors[1][:, 1]  # [:256]
        test_dataset = XYZ_DatasetWithIndices(test_x, test_y, z=get_color(test_x))
    else:
        assert False, 'BRO DO THIS'


    if args.train_filename is not None:
        train_dataset = _load_train_data_from_filename(args)

    print('DONE GETTING DATA OF SHAPES = train {}, test{}'.format( len(train_dataset), len(test_dataset)))

    if load:
        assert not isinstance(train_dataset, Subset)

    all_datasets = {
        'train': train_dataset,
        'test': test_dataset,
    }
    dataset_list_to_return = ['train', 'test']

    if eval_dataset is not None:
        all_datasets['eval'] = eval_dataset # this can be None at times
        dataset_list_to_return = ['train', 'test', 'eval']

    if return_datasets_only:
        LEN = len(train_dataset)
        perm = torch.randperm(LEN)
        m_train = int(0.8*LEN)
        train_indices = perm[:m_train]
        val_indices = perm[m_train:]
        all_datasets['split'] = {
            'train' : train_indices,
            'val' : val_indices
        }
        return all_datasets

    dataloader_dict = {}

    for key in dataset_list_to_return:
        SHUFFLE = DATA_HPARAMS['shuffle'] if key == "train" else False
        dataloader_dict[key] = DataLoader(
            all_datasets[key], batch_size=DATA_HPARAMS['batch_size'], shuffle=SHUFFLE, num_workers=DATA_HPARAMS['workers'], pin_memory=DATA_HPARAMS['pin_memory'], drop_last=False)

    return dataloader_dict