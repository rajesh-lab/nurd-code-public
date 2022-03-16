# utils.py

# utils.py

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Subset

waterbirds_list = ["waterbirds", "waterbirds_land", "waterbirds_water"]
known_group_list = ["joint", "chexpert", "mimic"] + waterbirds_list

def print_dict(dct):
    for item, val in dct.items():  # dct.iteritems() in Python 2
        print("{}: {:.3f}".format(item, val), end="\t"),
    print("")

def print_dict_of_dicts(dct_of_dcts):
    print("")
    for key, dct in dct_of_dcts.items():
        print(f" = {key} = ", end='\t')
        print_dict(dct)
        print("")

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
    def update(self, adict):
        self.__dict__.update(adict)

def get_DATA_HPARAMS(args):
    DATA_HPARAMS = {}
    DATA_HPARAMS['shuffle'] = True
    DATA_HPARAMS['subset'] = True
    DATA_HPARAMS['upsample'] = False
    DATA_HPARAMS['pin_memory'] = True
    DATA_HPARAMS['workers'] = args.workers
    DATA_HPARAMS['input_shape'] = [3 if args.color else 1] + \
        [args.img_side, args.img_side]
    DATA_HPARAMS['batch_size'] = args.nr_batch_size
    DATA_HPARAMS['hosp'] = args.hosp
    DATA_HPARAMS['rho'] = args.rho
    DATA_HPARAMS['rho_test'] = args.rho_test
    DATA_HPARAMS['label_balance_method'] = args.label_balance_method
    DATA_HPARAMS['change_disease'] = args.change_disease
    DATA_HPARAMS['grayscale'] = args.grayscale
    if 'naug' in vars(args).keys() and args.naug == "pr":
        DATA_HPARAMS['patch_size'] = args.patch_size

    return DATA_HPARAMS
    
def concat(vec_list):
    return torch.cat(vec_list, dim=0)


def add_uniform_weights_or_weighted_sample(args, DATA_HPARAMS, WEIGHT_LOADER_BATCHSIZE, loader=None, dataset=None, weights=None):
    # function to add a weights vector the dataloader; adds 1/|data size| weight to every sample;
    # this ensures that when I sum weighted results across the dataset; I get the correct dataset average of a metric
    assert loader is not None or dataset is not None
    if dataset is not None:
        _dataset = dataset
    else:
        _dataset = loader.dataset

    x = _dataset.tensors[0]
    y = _dataset.tensors[1].view(-1)

    if args.dataset == "joint":
        h = _dataset.tensors[2].view(-1)
    else:
        h = y.view(-1)
    w = torch.ones(y.shape).view(-1)/y.shape[0]
    uniformly_weighted_data = TensorDataset(
        x, y, h.to(y.device), w.to(y.device))
    if weights is None:
        return DataLoader(uniformly_weighted_data, batch_size=WEIGHT_LOADER_BATCHSIZE, num_workers=DATA_HPARAMS['workers'], pin_memory=DATA_HPARAMS['pin_memory'], drop_last=False, shuffle=False)
    else:
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=_dataset.tensors[0].shape[0], replacement=True)
        return DataLoader(uniformly_weighted_data, batch_size=WEIGHT_LOADER_BATCHSIZE, num_workers=DATA_HPARAMS['workers'], pin_memory=DATA_HPARAMS['pin_memory'], drop_last=False, shuffle=False, sampler=sampler)

def add_weights(loader, weights=None, return_dataset=False):
    assert not isinstance(loader.dataset, Subset)
    # function to add a weights vector the dataloader; adds 1/|data size| weight to every sample;
    # this ensures that when I sum weighted results across the dataset; I get the correct dataset average of a metric
    _dataset = loader.dataset
    _dataset.use_weights=True
    if weights is None:
        weights = torch.ones_like(_dataset.tensors[1]).float() # this is y
    _dataset.weights = weights/weights.shape[0]
    if return_dataset:
        return _dataset
    assert loader.dataset.use_weights
    assert loader.dataset.weights is not None
    return loader

def process_save_name(args):
    return _process_save_name_helper(
                            dataset=args.dataset,
                            change_disease=args.change_disease,
                            datasetTWO=args.datasetTWO,
                            rho=args.rho,
                            rho_test=args.rho_test,
                            img_side=args.img_side,
                            color=args.color,
                            true_waterbirds=args.true_waterbirds,
                            label_balance_method=args.label_balance_method,
                            eval=args.eval
                        )

def _process_save_name_helper(dataset, rho, rho_test, img_side, color, true_waterbirds=False, label_balance_method='downsample', eval=False, change_disease=None, datasetTWO="mimic",):
    if dataset == "joint":
        if change_disease is not None:
            if datasetTWO == "mimic":
                NAME = "joint_disease_{}_rho{}_rhotest{}".format(
                    change_disease, rho, rho_test)
            else:
                NAME = "joint_with_{}_disease_{}_rho{}_rhotest{}".format(
                    datasetTWO, change_disease, rho, rho_test)
        else:
            if datasetTWO == "mimic":
                NAME = "joint__rho{}__rhotest{}".format(rho, rho_test)
            else:
                NAME = "joint__{}_rho{}__rhotest{}".format(datasetTWO, rho, rho_test)
    else:

        NAME = "{}__rho{}__rhotest{}".format(
            dataset, rho, rho_test)

    if img_side != 224:
        NAME = "{}_SIDE{}".format(NAME, img_side)

    NAME = "{}_color{}".format(NAME, color)

    if true_waterbirds and dataset == "waterbirds":
        NAME = "{}_trueWaterbirds".format(NAME)
    
    NAME = "{}_balance{}".format(NAME, label_balance_method)

    NAME = "{}_eval{}".format(NAME, eval)

    if dataset == "waterbirds":
        NAME = "{}_fixedsplit".format(NAME)
    
    if dataset == "synthetic":
        NAME = "{}_large".format(NAME)

    return NAME


def _WEIGHT_MODEL_NUISANCE_TYPE(args):
    if args.naug:
        return args.naug
    if args.zero_nuisance:
        return "zero_nuisance"
    
    if args.dataset == "cmnist":
        return "cmnist_color"
    elif args.dataset == "synthetic" :
        return "synthetic_last"
    elif args.dataset == "joint":
        return "holemask" #side_and_top_mask"
    else:
        return "holemask"

def _PRED_MODEL_NUISANCE_TYPES(args):

    weight_model_nuisance_type = _WEIGHT_MODEL_NUISANCE_TYPE(args)

    if args.pred_from_center:
        print("PREDICTION FROM CENTER.")
        print("PREDICTION FROM CENTER.")
        print("PREDICTION FROM CENTER.")
        print("PREDICTION FROM CENTER.")
        return weight_model_nuisance_type, "onlycenter"

    if args.naug:
        return weight_model_nuisance_type, "identity"
    if args.zero_nuisance:
        return weight_model_nuisance_type, "identity"

    if args.dataset == "cmnist":
        return weight_model_nuisance_type, "identity"
    elif args.dataset == "synthetic":
        return weight_model_nuisance_type, "synthetic_first_two"
    elif args.dataset == "joint":
        return weight_model_nuisance_type, "identity"
    else:
        return weight_model_nuisance_type, "identity"

all_params = [
    'config',
    'logging',
    'dataset',
    'model',
    'training',
    'optimization',
    'nuisance'
]

def add_args(parser, parameter_type):
    if parameter_type == "config":
        # generic parameters
        parser.add_argument('--prefix', type=str, help='folder organization', default=None)
        parser.add_argument('--add_pred_suffix', type=str, help='folder organization for the pred models', default=None)
        parser.add_argument('--add_final_results_suffix', type=str, help='folder organization for the pred models', default=None)

        parser.add_argument('--seed', type=int, default=1234, help='seed for the run')
        parser.add_argument('--workers', type=int, default=2, help='Number of workers to use in loaders')
        parser.add_argument('--debug',  type=int, default=0, help='print things; at 1 print only pred model stuff, at 2 print everything')
        parser.add_argument('--device', type=str, default="", help='DONT CHANGE')
        parser.add_argument('--nr_only', action='store_true', help='run weight models only and stop')
        parser.add_argument('--load_weights', action='store_true', help='load weights and do dist')
        parser.add_argument('--eval', action='store_true', help='uses datasets with eval subsets')
        parser.add_argument('--load_dist_model', action='store_true', help='loads saved distillation model and only report perf; no training')
        parser.add_argument('--load_final_model', action='store_true', help='loads saved final model and only report perf; no training')
        parser.add_argument('--dont_do_critic', action='store_true', help='prevents the critic training on loaded model', default=False)
        parser.add_argument('--dont_save_final_results', action='store_true', help='prevents saving the last models', default=False)
        parser.add_argument('--warmup_critic', action='store_true', help='runs an initial warmup for critic before epoch 0 to help get better signals. Not allowed with randomrestart > 0.', default=False)


    if parameter_type == "logging":
        # logging parameters
        parser.add_argument('--print_interval', type=int, default=10, help='mini batch size')

    if parameter_type == "dataset":
        # dataset_parameters
        parser.add_argument('--dataset', type=str, default="chexpert", help='Which dataset to do ERM')
        parser.add_argument('--true_waterbirds', action='store_true', help="true waterbirds")
        parser.add_argument('--img_side', type=int, default=32, help='run on images of size')
        parser.add_argument('--dry', action='store_true', help='run on subset')
        parser.add_argument('--train_filename', type=str, default=None, help='use this dataset instead')
        parser.add_argument('--hosp_predict', action='store_true', help='Predict hospital from images')
        parser.add_argument('--datasetTWO', type=str, default="mimic", help='use this dataset along with chexpert to construct joint')
        parser.add_argument('--rho', type=float, default=0.9, help='value or correlation for the joint dataset')
        parser.add_argument('--rho_test', type=float, default=0.9, help='value or correlation in the test data. for all data')
        parser.add_argument('--rho_test_for_eval', type=float, default=None, help='only to be used in evaluation where the test data can be changed from what is loaded using this argument.')
        #  (TODO:make sure train and test datasets can be loaded separately; this argument can be removed then.)

        parser.add_argument('--label_balance_method', type=str, default="downsample", help='upsample or downsample to balance in the joint dataset', choices=["upsample", "downsample"])
        parser.add_argument('--equalized_groups', action='store_true', help="equal props")
        parser.add_argument('--change_disease', type=str, help='disease to use instead of pneumonia', default=None)

        parser.add_argument('--color', action='store_true', help='set images to color')    
        parser.add_argument('--grayscale', action='store_true', help='set images to grayscale')

        parser.add_argument('--hosp', action='store_true', help='prediction only from the center', default=True)

    if parameter_type == "model":
        # model parameters
        parser.add_argument('--pred_model_type', type=str, default=None, help='pred model type')
        parser.add_argument('--weight_model_type', type=str, help='weight model type', required=True)
        parser.add_argument('--nr_epochs', type=int, default=10, help='mini batch size')
        parser.add_argument('--dist_epochs', type=int, default=10, help='mini batch size')
        # parser.add_argument('--ensemble_count', type=int, default=1, help='Number of models to average in the weights model')
        parser.add_argument('--num_folds', type=int, default=2, help='Number of models to average in the weights model')
        # parser.add_argument('--threshold', type=float, default=-1, help='break weights')
        # parser.add_argument('--platt', action='store_true', help="do 'platt' scaling by logistic regression")

    if parameter_type == "training":
        # training parameters
        parser.add_argument('--nr_batch_size', type=int, default=128, help='mini batch size')
        parser.add_argument('--dist_batch_size', type=int, default=256, help='batchsize for represntation learning minimax')
        parser.add_argument('--noise', action='store_true', help='add training noise to x')
        parser.add_argument('--coeff', type=float, default=-1, help='Specified probability mass for y,z')
        parser.add_argument('--exact', action='store_true', help='use exact weights')
        parser.add_argument('--lambda_', type=float, default=0.0, help='representation regularization loss')
        parser.add_argument('--max_lambda_', type=float, default=0, help='representation regularization max')
        parser.add_argument('--frac_phi_steps', type=float, default=0.0, help='fraction on a epoch that p_phi steps')
        # parser.add_argument('--randomrestart', action='store_true', help="restarts training of the phi model before every gamma theta step")
        parser.add_argument('--randomrestart', type=int, default=-1, help="if K, restarts training of the phi model before every K gamma theta step. <0 disables.")
        parser.add_argument('--conditional', action='store_true', help="impose conditional independence only, otherwise this does joint independence")

    if parameter_type == "optimization":
        # optimization arguments
        default_lr=1e-3
        parser.add_argument('--theta_lr', type=float, default=default_lr, help='learning rate for theta in p_theta(Y | r_gamma(X) ) ')
        parser.add_argument('--gamma_lr', type=float, default=default_lr, help='learning rate for the representation')
        parser.add_argument('--phi_lr', type=float, default=default_lr, help='learning rate for phi in  p_phi(Y | r_gamma(X), Z ) ')

        parser.add_argument('--dist_decay', type=float, default=1e-4, help='l2 regularization for predictive model and representation')
        parser.add_argument('--phi_decay', type=float, default=1e-4, help='l2 regularization only works critic model')

        parser.add_argument('--augment', action='store_true', help="augment every batch half the time")

        parser.add_argument('--nr_lr', type=float, default=default_lr, help='learning rate for theta in p_theta(Y | r_gamma(X) ) ')
        parser.add_argument('--nr_decay', type=float, default=1e-4, help='l2 regularization for the nuisance-randomization stage')
        parser.add_argument('--nr_strategy', type=str, default="weight", help="upsample, weight, or downsample", choices=["upsample", "downsample", "weight"])
        parser.add_argument('--disc_groups', type=int, default=2, help="disc groups")

    if parameter_type == "nuisance":
        # nuisance specification 
        parser.add_argument('--naug', type=str, default=None, help="type of naug", choices=["HG", "PR", "hybrid"])
        parser.add_argument('--patch_size', type=int, default=None, help='patch size in randomization')
        parser.add_argument('--sigma', type=float, default=None, help="add noise to the image")
        parser.add_argument('--border', type=float, default=7, help='value or correlation in the test data. for all data')
        parser.add_argument('--zero_nuisance', action='store_true', help='0 * nuisance to nurd')
        parser.add_argument('--pred_from_center', action='store_true', help=' prediction only from the center ')


def SANITY_CHECKS_ARGS(args):
    assert not args.color or not args.grayscale

    if args.naug is not None:
        if args.naug == "HG":
            assert args.sigma is not None

        if args.naug == "PR":
            assert args.patch_size is not None
    
    assert args.lambda_ > -1e-3, args.lambda_
    assert not args.conditional, "THIS CODE HAS NOT BEEN CHECKED TO WORK."

    # if args.dataset == "synthetic":
    # assert not args.load_weights, "CANNOT LOAD WEIGHTS BECAUSE THE SYNTHETIC DATA IS GENERATED IN EACH CALL TO THE DATALOADERS."
    
    assert args.naug is None, "COMMENT THIS TO RUN WITH NEGATIVE NUISANCES."

    if args.warmup_critic:
        assert args.randomrestart < 0, args.randomrestart
    # if args.dont_do_critic:
    # assert args.load_dist_model or args.load_final_model
    # assert not args.nr_only or not args.load_weights, "why load weights if you want to run nuisance randomization only"

def get_color(batch_x):
    c = torch.max(batch_x + 1, dim=2)[0]
    c = c.max(dim=2)[0]
    c = c.argmax(dim=1)
    batch_c = c.view(-1, 1, 1, 1)
    return batch_c.view(-1)

# def get_selectors(y_long, c_long, coeff):
#     y_1s = (y_long == 1).int().view(-1)
#     c_1s = (c_long == 1).int().view(-1)

#     # assuming only 0, 1 in these; TODO: generalize
#     assert coeff > 0, coeff
#     selectors = {
#         '00' : [coeff, (1-y_1s)*(1 - c_1s)],
#         '10' : [1 - coeff , y_1s*(1 - c_1s)],
#         '01' : [1 - coeff, (1-y_1s)*c_1s],
#         '11' : [coeff,  y_1s*c_1s]
#     }
#     return selectors

# def get_weights(y, z, exact, y_weights, coeff, dataset, weight_func, h=None):
#     y_long = y.long()    
#     if exact:
#         assert dataset != "synthetic"
#         if dataset=="cmnist":
#             c = get_color(x, y)
#         elif dataset in known_group_list:
#             c = h.view(-1).long()
#         elif dataset in ["synthetic"]:
#             c = x[:,2].long() #>0.0).long() #
#         else:
#             assert False, "BRO DATASET FAIL"
        
#         c = c.to(y.device)

#         c_long = c.long()
#         selectors = get_selectors(y_long, c_long, coeff)

#         mass = 0*y.float()
#         for key, val in selectors.items():
#             mass[val[1] > 0] = val[0]

#         for key, y_mass in y_weights.items():
#             mass[y_long==key] = mass[y_long==key]/y_mass # divides by p(Y=k); inversion later.

#         return 1/mass
#     else:   
#         raise NotImplementedError("THIS FUNCTIONALITY MOVED TO THE MAIN FILE.")


def multiply_by_marginal_of_y(y_long, weights):
    for key in torch.unique(y_long):
        y_mass = torch.mean((y_long==key).float())
        weights[y_long==key] = y_mass/weights[y_long==key]
    return weights

def multiple_by_labelmarginal_and_balance_weights(weight_vec, y_long, y_weights_from_array=None):
    """produce balanced weights to use in distillation from weights that are 1/p(Y | Z)

    Args:
        weight_vec (_type_): unnormalized weights of the type 1/p(Y|Z)
        y_long (_type_): the y vector for the whole dataset

    Returns:
        _type_: weights of the same type as _weight_vec; this is also an in_place operation.
    """
    _weights = weight_vec # multiply_by_marginal_of_y(weight_vec, y_long)
    for key in torch.unique(y_long):
        if y_weights_from_array is None:
            y_mass = torch.mean((y_long==key).float()) # compute the probability p(Y = key)
        else:
            y_mass = torch.mean((y_weights_from_array==key).float())

        # this multiplies each weight by the marginal of Y=key, to get P(Y = key) / p(Y = key | Z)
        _weights[y_long==key] = y_mass*_weights[y_long==key] 
        # this ensures that the marginal probability of the weighted distribution is the same as before weighting : p_W(Y = key) = p(Y=key)
        _weights[y_long==key] = y_mass*_weights[y_long==key]/torch.sum(_weights[y_long==key])

        # old_code
        # _weights[y_long==1] = torch.mean((y_long==1).float())*_weights[y_long==1]/torch.sum(_weights[y_long==1])
        # _weights[y_long==0] = torch.mean((y_long==0).float())*_weights[y_long==0]/torch.sum(_weights[y_long==0])

    total_mass_sum = torch.sum(_weights)
    _weights = _weights/total_mass_sum

    # print(" AFTER MARGINAL WEIGHT FIXING ")
    # print(" 1 weight ", torch.sum(_weights*y_long))
    # print(" 0 weight ", torch.sum(_weights*(1 - y_long)))

    return _weights

def make_weights(y, z, args, weights_type="balanced"):
    assert weights_type in ['balanced', 'extreme'], weights_type
    if args.dataset == "synthetic":
        print("ERM IS THE ONLY CASE FOR WHICH THE SYNTHETIC DATASET WILL REQUIRE WEIGHTS. THIS IS BECAUSE Z is CNTS IN THE SYNTHETIC EXAMPLE.")
        assert abs(args.coeff - 0.5) < 1e-3, args.coeff
        return args.coeff*torch.ones_like(y).float().view(-1)
    weights = torch.zeros_like(y).float()
    y_long = y.long().view(-1)
    z_long = z.long().view(-1)
    if args.coeff > 0:
        assert y.max() < 2, y.max()
        assert z.max() < 2, z.max() # the coeff specification is only built for
        weights[y_long==z_long] = 1/args.coeff
        weights[y_long!=z_long] = 1/(1 - args.coeff)

        for y_key in torch.unique(y_long):
            for z_key in torch.unique(z_long):
                y_check = y_long==y_key
                z_check = z_long==z_key
                yz_key_slice = torch.logical_and(y_check, z_check)
                print(y_key, z_key, weights[yz_key_slice].mean())
    else:
        for y_key in torch.unique(y_long):
            for z_key in torch.unique(z_long):
                y_check = y_long==y_key
                z_check = z_long==z_key

                p_ygz_estimate = z_check.float().mean()/(z_check.float()*y_check.float()).mean() # this is p(z) / p(y,z) = 1 / p(y | z)

                yz_key_slice = torch.logical_and(y_check, z_check)
                weights[yz_key_slice] = p_ygz_estimate
                print(y_key, z_key, p_ygz_estimate)
                
    
    if weights_type=="extreme":
        # creating extreme weights
        prob_masses = 1/weights
        # .5/
        assert prob_masses.max() < 1, (prob_masses.min(), prob_masses.max())
        assert prob_masses.min() > 0, (prob_masses.min(), prob_masses.max())
        masses_gt05 = prob_masses > 0.5
        masses_lt05 = prob_masses <= 0.5

        extreme_mass = weights.float()*0
        extreme_mass[masses_gt05] = 1 - 0.5*(1 - weights[masses_gt05]) # values bigger than 0.5 gets closer to 1
        extreme_mass[masses_lt05] = 0.5*weights[masses_lt05] # values smaller than 0.5 get closer to 0

        # print("===========")
        extreme_weights = 1/extreme_mass.cpu()
        return extreme_weights.view(-1) # the idx helps make sure that the weights are assigned to samples correctly 
    else:
        return weights.view(-1) # the idx helps make sure that the weights are assigned to samples correctly 

def get_random_threeway_split(list_to_split, size1, size2, size3):
    perm = torch.randperm(list_to_split.shape[0])
    permed_subset = list_to_split[perm][:size1 + size2 + size3]
    return permed_subset[:size1], permed_subset[size1:size1+size2], permed_subset[size1+size2:]


def get_random_split(list_to_split, size1, size2):
    perm = torch.randperm(list_to_split.shape[0])
    permed_subset = list_to_split[perm][:size1 + size2]
    return permed_subset[:size1], permed_subset[size1:]

# train split into train/val

def split_20percent_array(array):
    m = len(array)
    m_train = int(0.8*m)
    return array[:m_train], array[m_train:]


def random_subset_of_array(_dummy_dataset, MAX_KEEP):
    LEN = len(_dummy_dataset)
    perm = torch.randperm(LEN)
    KEEP_INDICES = perm[:MAX_KEEP]
    return Subset(_dummy_dataset, KEEP_INDICES)


def get_random_subset(_dummy_dataset, total_size, size1, size2):
    perm = torch.randperm(total_size)
    permed_subset = perm[:size1 + size2]
    return Subset(_dummy_dataset, permed_subset[:size1]), Subset(_dummy_dataset, permed_subset[size1:])

def get_index_subset(train_group_sizes, test_group_sizes, waterbirds_data):
    train_group_indices = []
    test_group_indices = []
    for group_id in [0, 1, 2, 3]:
        group_id_indices = (torch.from_numpy(
            waterbirds_data.dataset.group_array).long() == group_id).nonzero()
        train_indices, test_indices = get_random_split(
            group_id_indices, train_group_sizes[group_id], test_group_sizes[group_id])

        # print(group_id, train_indices.shape, test_indices.shape)

        train_group_indices.append(train_indices.view(-1))
        test_group_indices.append(test_indices.view(-1))

        print("getting to group_id : {} with size {}".format(
            group_id, group_id_indices.shape))

    return train_group_indices, test_group_indices