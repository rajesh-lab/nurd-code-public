# Reweighting-NURD Copyright (c) aahlad puli
#
# TODO: set arguments for example run and save somewhere

import os
import copy
import random
from re import M
from black import diff
import numpy as np

import torch
from torch.utils.data import Subset
from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import matplotlib.pyplot as plt
import crossval 
import training_functions
import utils

from dataloaders import construct_dataloaders, create_loader
from tensorboardX import SummaryWriter
from utils import known_group_list, waterbirds_list

# device = "cuda"

_SAVEFOLDER = "LOGS"
_SAVEFOLDER_MODELS = "SAVED_MODELS"
EPS=1e-4

def cli_main():

    # TODO: training_functions too big
    # TODO: need to make sure synthetic is creating the right dataset

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
    # parser.add_argument('--rho_test_for_eval', type=float, default=None, help='only to be used in evaluation where the test data can be changed from what is loaded using this argument.')
        #  (TODO:make sure train and test datasets can be loaded separately; this argument can be removed then.)
    args = parser.parse_args()
    assert args.prefix is not None, "args.prefix is required."
    assert args.add_final_results_suffix is not None, "when evaluating, the save space should be specified because things can get overwritten otherwise."
    
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

    # this type of dataset has one variables used within training for dist, weights. this is set by the code, so don't make it protected
    train_dataset = datasets_dict['train']

    print("-------------------------------")

    # WEIGHT MODEL
    # WEIGHT MODEL
    # WEIGHT MODEL
    # WEIGHT MODEL
    
        
    if args.exact:
        print(" SKIPPING WEIGHT MODEL ")
        nr_stage_results_dict = {}
        weights = utils.make_weights(train_dataset.y, train_dataset.z, args=args)
    else:
        weights_save_pt_filename = "{}/weights_{}.pt".format(_SAVEFOLDER, args.prefix)
        assert os.path.isfile(weights_save_pt_filename)
        print(" LOADING WEIGHTS IN {}".format(weights_save_pt_filename))

        weights_saved_dict = torch.load(weights_save_pt_filename)
        weights = weights_saved_dict['weights']
        nr_stage_results_dict = weights_saved_dict['nr_stage_results_dict']
    
    weights = weights.cpu()

    PROB_MIN = 0.05
    MAX_CLAMP = 1/PROB_MIN # 20
    MIN_CLAMP = 1/(1-PROB_MIN) # 1.05
    # assert False, (MIN_CLAMP, MAX_CLAMP)
    weights = weights.clamp(max=MAX_CLAMP, min=MIN_CLAMP)

    _split = datasets_dict['split']
    train_split = _split['train']
    val_split = _split['val']

    train_weights = weights[train_split]
    val_weights = weights[val_split]

    y_train_long = train_dataset.y[train_split].long().view(-1).to(weights.device)
    y_val_long = train_dataset.y[val_split].long().view(-1).to(weights.device)
    y_test_long = datasets_dict['test'].y.long().view(-1).to(weights.device)

    weights[train_split] = utils.multiple_by_labelmarginal_and_balance_weights(train_weights, y_train_long, y_weights_from_array=y_test_long)
    weights[val_split] = utils.multiple_by_labelmarginal_and_balance_weights(val_weights, y_val_long, y_weights_from_array=y_test_long)

    # get a copy of the dataset to create an unbalanced validation dataset
    # unbalanced_train_dataset = copy.deepcopy(train_dataset) # this will 
    # unbalanced_train_dataset.weights = torch.zeros_like(unbalanced_train_dataset.y.view(-1)) # create all 0 weights
    # equal_val_weights = torch.ones_like(val_weights.view(-1)) # only create validation weights 
    # unbalanced_train_dataset.weights[val_split] = utils.multiple_by_labelmarginal_and_balance_weights(equal_val_weights, y_val_long, y_weights_from_array=y_test_long) # balanced validation weights
    # equal_val_weights_for_checking = unbalanced_train_dataset.weights[val_split]
    # assert equal_val_weights_for_checking.max()>0, equal_val_weights_for_checking.max()
    # assert (equal_val_weights_for_checking.max() - equal_val_weights_for_checking.min()).abs()/equal_val_weights_for_checking.max() < 1e-1, (equal_val_weights_for_checking.max(), equal_val_weights_for_checking.min()) # check weights are equal

    # add weights to the train dataset, then subset and create loaders
    train_dataset.weights = weights

    DIST_LOADER_PARAMS = {}
    DIST_LOADER_PARAMS['BATCH_SIZE'] = args.dist_batch_size
    DIST_LOADER_PARAMS['workers'] = args.workers
    DIST_LOADER_PARAMS['pin_memory'] = True

    weighted_train_loader = create_loader(
        Subset(train_dataset, train_split),
        DIST_LOADER_PARAMS,
        shuffle=True,
        weights=weights[train_split], # this is used to produce a sampler because we cannot access weights from subset easily.
        strategy=args.nr_strategy
    )

    weighted_train_loader_for_evaluation_only = create_loader(
        Subset(train_dataset, train_split),
        DIST_LOADER_PARAMS,
        shuffle=False,
        strategy="weight"
    )

    weighted_val_loader = create_loader(
        Subset(train_dataset, val_split),
        DIST_LOADER_PARAMS,
        shuffle=False,
        strategy="weight"
    )

    # unbalanced_val_loader = create_loader(
    #     DIST_LOADER_PARAMS,
    #     shuffle=False,
    #     strategy="weight"
    # )


    
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
        'unbal_val' : create_loader(Subset(train_dataset, val_split), DIST_LOADER_PARAMS), # keys like unbal_val will make sure that the weights are not used 
        'test': create_loader(datasets_dict['test'], DIST_LOADER_PARAMS) # we compute unweighted avg (plain_avg) for 'test' and 'eval' key in this dict
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

    # assert False, pred_model_bunch.device
    if args.load_dist_model:
        # dist_model = final_saver.load_best(map_location=pred_model_bunch.device, objective='dist')
        dist_model = dist_saver.load_best(map_location=pred_model_bunch.device, objective='dist')
        print("LOADED PRE-CRITIC DISTILLATION MODEL.")
    elif args.load_final_model:
        dist_model = final_saver.load_best(map_location=pred_model_bunch.device, objective='dist')
        print("LOADED POST-CRITIC DISTILLATION MODEL.")
    else:
        assert False, "you need to load some model."
    
    training_functions.PUT_MODELS_ON_DEVICE(dist_model, args.device)

    final_results = evaluate_model_on_dist_loaders(dist_model)
    utils.print_dict_of_dicts(final_results)
    print("")

    # ------------
    # conluding, printing, and saving
    # ------------

    if not args.dont_save_final_results:
        results_pt_filename = "{}/evaluations_{}{}_{}.pt".format(_SAVEFOLDER, args.prefix, "" if args.add_pred_suffix is None else args.add_pred_suffix, args.add_final_results_suffix)
        print(" SAVING RESULTS IN {}".format(results_pt_filename))
        utils.print_dict_of_dicts(final_results)

        torch.save(
            {
                'nr_stage_results_dict': nr_stage_results_dict,
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
