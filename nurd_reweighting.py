# Reweighting-NURD Copyright (c) aahlad puli
#
# TODO: set arguments for example run and save somewhere

import os
import copy
import random
from re import M
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
    args = parser.parse_args()
    assert args.prefix is not None, "args.prefix is required."
    assert args.rho_test_for_eval is None, "this will overwrite the test set being loaded; please use the evaluation script if you want to just evaluate. you can also set rho_test."
    
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
        if os.path.isfile(weights_save_pt_filename) and args.load_weights:
            print(" LOADING WEIGHTS IN {}".format(weights_save_pt_filename))

            weights_saved_dict = torch.load(weights_save_pt_filename)
            weights = weights_saved_dict['weights']
            
            nr_stage_results_dict = weights_saved_dict['nr_stage_results_dict']
        else:
            print(" ESTIMATING WEIGHTS; WILL BE SAVED IN {}".format(weights_save_pt_filename))
            # weight model setup
            nr_stage_results_dict = {}

            weight_model_transform_type = utils._WEIGHT_MODEL_NUISANCE_TYPE(args)
            weight_model_type = args.weight_model_type

            weight_model_hparams = dict(model_type=weight_model_type,
                                        weight_model=True,
                                        model_stage="weight_model",
                                        transform_type=weight_model_transform_type,
                                        verbose=args.debug - 1, in_size=3,
                                        mid_dim=16,
                                        out_dim=1
                                    )

            weight_model_hparams.update(vars(args))
            weight_model_bunch = utils.Bunch(weight_model_hparams)

            # train val split

            proba_list = []
            idx_list = []

            LOADER_HPARAMS = {   
                'BATCH_SIZE' : args.nr_batch_size,
                'workers' : args.workers,
                'pin_memory' : True
            }

            for fold, (train_idx, val_idx, test_idx) in enumerate(crossval.k_folds(n_splits=args.num_folds, subjects=len(train_dataset))):
                # some sanity checks
                # some sanity checks
                # some sanity checks
                # assert len(np.intersect1d(train_idx, val_idx)) == 0, np.intersect1d(train_idx, val_idx)

                if args.debug >= 2:
                    # THIS ASSUMES THAT THE TRAIN DATASET HAS A BINARY Z.
                    print("SANITY CHECKS")
                    y_local = train_dataset.y
                    z_local = train_dataset.z

                    print("     Y/Z E and NE = {}/{}".format(y_local[y_local == z_local].shape, y_local[y_local != z_local].shape))

                    y_local_train = y_local[train_idx]
                    y_local_val = y_local[val_idx]

                    z_local_train = z_local[train_idx]
                    z_local_val = z_local[val_idx]

                    print("     Y/Z full mean = {:.3f}/{:.3f}".format(y_local.float().mean(), z_local.float().mean()))                   

                    print("     Y shapes = {}/{}".format(y_local_train.shape, y_local_val.shape))
                    print("     Y means = {:.3f}/{:.3f}".format(y_local_train.float().mean(), y_local_val.float().mean()))

                    print("     BIAS full means = {:.3f}".format((z_local==y_local).float().mean()))
                    print("     BIAS means = {:.3f}/{:.3f}".format((z_local_train==y_local_train).float().mean(), (z_local_val==y_local_val).float().mean()))

                # some sanity checks
                # some sanity checks

                z_train_loader = create_loader( Subset(train_dataset, indices=train_idx), LOADER_HPARAMS, shuffle=True)
                z_val_loader = create_loader( Subset(train_dataset, indices=val_idx), LOADER_HPARAMS, shuffle=False)
                z_test_loader = create_loader( Subset(train_dataset, indices=test_idx), LOADER_HPARAMS, shuffle=False)

                print("----------------------------------------------")
                print(" ---- RUNNING WEIGHT MODEL {}/{}".format(fold+1,args.num_folds))
                print("")
                WEIGHTSAVEDIR = "{}/weightmodel{}_{}".format(_SAVEFOLDER_MODELS, fold, weight_model_bunch.prefix)
                nr_saver = training_functions.ModelSaver(
                                save_dir=WEIGHTSAVEDIR,
                                args=weight_model_bunch
                            )
                
                weight_model, final_results = training_functions.fit_model(
                    args_bunch=weight_model_bunch,
                    epochs=weight_model_bunch.nr_epochs,
                    loaders={
                        'train': z_train_loader,
                        'val': z_val_loader,
                        'test' : z_test_loader
                    },
                    objective='erm',
                    writer=SummaryWriter(logdir=WEIGHTSAVEDIR),
                    saver=nr_saver
                )

                training_functions.PUT_MODELS_ON_DEVICE(weight_model, weight_model_bunch.device)
                print("predict probabilities")
                proba = training_functions.predict_proba(z_test_loader, weight_model, weight_model_bunch)
                proba = proba.cpu()
                proba_list.append(proba.view(-1))
                idx_list.append(test_idx.view(-1))
                nr_stage_results_dict[f"results_{fold}"] = final_results
                nr_stage_results_dict[f"modelpath_{fold}"] = nr_saver.best_path
                del weight_model
                    
            # the weights are the inverse of the concatenated lists of  probabilities for each validation loader place at the right index w.r.t. the true dataset
            proba = utils.concat(proba_list)
            idx = utils.concat(idx_list)
            weights = torch.zeros_like(proba)
            weights[idx] = 1/(EPS + proba)
    
            # BY THIS TIME WEIGHTS ESTIMATES 1/P(Y|Z)
            print(" WEIGHT ESTIMATION TRAINING DONE.")
            print("----------------------------------------------")
            print("----------------------------------------------")

        # THIS IS JUST FOR SUMMARIZATION
        results_accumulator = {}
        for fold in range(args.num_folds):
            val_results = nr_stage_results_dict[f"results_{fold}"]
            # utils.print_dict_of_dicts(val_results)
            for key, val in val_results.items():
                results_accumulator[key] = results_accumulator.get(key,{})
                results_accumulator[key]['pred_loss'] = results_accumulator[key].get('pred_loss',0) + val['pred_loss']/args.num_folds
                results_accumulator[key]['acc'] = results_accumulator[key].get('acc',0) + val['acc']/args.num_folds
            
        utils.print_dict_of_dicts(results_accumulator)

    weights = weights.cpu()
    # BALANCE THE WEIGHTS HERE AND ALSO MATCH THE WEIGHT FOR EACH CLASS EQUAL ITS MARGINAL PROBABILITY
    if not args.exact:
        # we only save for learned weights
        print(" SAVING WEIGHTS IN {}".format(weights_save_pt_filename))

        torch.save(
            {
                'weights': weights,
                'nr_stage_results_dict': nr_stage_results_dict,
            },
            weights_save_pt_filename
        )

    PROB_MIN = 0.05
    MAX_CLAMP = 1/PROB_MIN # 20
    MIN_CLAMP = 1/(1-PROB_MIN) # 1.05
    # assert False, (MIN_CLAMP, MAX_CLAMP)
    weights = weights.clamp(max=MAX_CLAMP, min=MIN_CLAMP)

    _split = datasets_dict['split']
    train_split = _split['train']
    val_split = _split['val']

    if args.debug == 10:
        # THIS ASSUMES THAT THE TRAIN DATASET HAS A BINARY Z.
        # the function ends here
        # z_long = train_dataset.z.long()
        # y_long = train_dataset.y.long()

        print("PLOTTING AND WEIGHT SANITY CHECKS")
        true_weights = utils.make_weights(train_dataset.y, train_dataset.z, args=args).cpu()
        print(weights[train_split].shape, weights[val_split].shape)
        plt.figure()
        plt.hist(1/weights.cpu().numpy(), bins=40)
        plt.savefig("./FIGURES/{}_est_hist.png".format(args.prefix), dpi=150)
        plt.figure()
        diff_weights = true_weights - weights
        print("MEAN", ((diff_weights/true_weights).abs() < 0.10).float().mean())
        plt.hist(diff_weights.numpy(), bins=200)
        plt.xlim([-10, 10])
        plt.savefig("./FIGURES/{}_diff_hist.png".format(args.prefix), dpi=150)
        plt.figure()
        plt.scatter(1/true_weights.cpu().numpy(), 1/weights.cpu().numpy())
        plt.savefig("./FIGURES/{}_est_v_true.png".format(args.prefix), dpi=150)

        y_local = train_dataset.y.view(-1).long()
        z_local = train_dataset.z.view(-1).long()
        
        matches_prob = 1/weights[y_local == z_local]
        mismatches_prob = 1/weights[y_local != z_local]

        print(matches_prob.max(), matches_prob.min())
        print(mismatches_prob.max(), mismatches_prob.min())

        plt.figure()
        plt.hist(matches_prob.view(-1).numpy(), bins=50, label="majority", density=True, alpha=0.5)
        plt.hist(mismatches_prob.view(-1).numpy(), bins=50, label="minority", density=True, alpha=0.5)
        plt.legend()
        plt.savefig("./FIGURES/{}_maj_v_min.png".format(args.prefix), dpi=150)

        prob_z1 = 1/weights[z_local==1]
        prob_z0 = 1/weights[z_local==0]

        plt.figure()
        plt.hist(prob_z1.view(-1).numpy(), bins=50, label="chexpert", density=True, alpha=0.5)
        plt.hist(prob_z0.view(-1).numpy(), bins=50, label="mimic", density=True, alpha=0.5)
        plt.legend()
        plt.savefig("./FIGURES/{}_z0_v_z1.png".format(args.prefix), dpi=150)


        prob_z1_y1= 1/weights[torch.logical_and(z_local==1, y_local==1)]
        prob_z0_y1= 1/weights[torch.logical_and(z_local==0, y_local==1)]

        prob_z1_y0 = 1/weights[torch.logical_and(z_local==1, y_local==0)]
        prob_z0_y0= 1/weights[torch.logical_and(z_local==0, y_local==0)]


        for arr in [
                prob_z1_y1 ,prob_z0_y1 ,prob_z1_y0 ,prob_z0_y0
        ]:
            print(arr.shape, arr.max(), arr.min(), arr.mean())

        plt.figure()
        plt.hist(prob_z0_y1.view(-1).numpy(), bins=50, label="z=0,y=1", alpha=0.5)
        plt.hist(prob_z0_y0.view(-1).numpy(), bins=50, label="z=0,y=0", alpha=0.5)
        plt.legend()
        plt.savefig("./FIGURES/{}_groups_mimic.png".format(args.prefix), dpi=150)

        plt.figure()
        plt.hist(prob_z1_y1.view(-1).numpy(), bins=50, label="z=1,y=1", alpha=0.5)
        plt.hist(prob_z1_y0.view(-1).numpy(), bins=50, label="z=1,y=0", alpha=0.5)
        plt.legend()
        plt.savefig("./FIGURES/{}_groups_chexpert.png".format(args.prefix), dpi=150)

        return
    
    train_weights = weights[train_split]

    y_train_long = train_dataset.y[train_split].long().view(-1).to(weights.device)
    y_val_long = train_dataset.y[val_split].long().view(-1).to(weights.device)
    y_test_long = datasets_dict['test'].y.long().view(-1).to(weights.device)

    if args.bal_val:
        print(" Exactly balancing the validation data. ")
        z_val_long = train_dataset.z[val_split].long().view(-1).to(weights.device)
        assert y_val_long.shape == z_val_long.shape, (y_val_long.shape, z_val_long.shape)
        val_weights = utils.make_weights(y_val_long, z_val_long, args=args, ignore_coeff=args.bal_val)
        # assert False, (val_weights[y_val_long == z_val_long][:10], val_weights[y_val_long != z_val_long][:10])
    else:
        val_weights = weights[val_split]

    weights[train_split] = utils.multiple_by_labelmarginal_and_balance_weights(train_weights, y_train_long, y_weights_from_array=y_test_long)
    weights[val_split] = utils.multiple_by_labelmarginal_and_balance_weights(val_weights, y_val_long, y_weights_from_array=y_test_long)

    if args.nr_only:
        if args.load_weights:
            return
        print("SKIPPING DISTILLATION; ADD --load_weights to use the saved weights and run distillation.")
        return # skips the rest
    
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
        'test': create_loader(datasets_dict['test'], DIST_LOADER_PARAMS),
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

    if args.load_dist_model:
        # dist_model = final_saver.load_best(map_location=pred_model_bunch.device, objective='dist')
        dist_model = dist_saver.load_best(map_location=pred_model_bunch.device, objective='dist')
        print("LOADED PRE-CRITIC DISTILLATION MODEL.")
    elif args.load_final_model:
        dist_model = final_saver.load_best(map_location=pred_model_bunch.device, objective='dist')
        print("LOADED POST-CRITIC DISTILLATION MODEL.")
    else:
        dist_model, _ = training_functions.fit_model(
            args_bunch=pred_model_bunch,
            epochs=pred_model_bunch.dist_epochs,
            loaders=dist_loaders,
            objective='dist',
            writer=SummaryWriter(logdir=DISTSAVEDIR),
            saver=dist_saver
        )

        print("DISTILLATION COMPLETE.")
        
    training_functions.PUT_MODELS_ON_DEVICE(dist_model, args.device)

    dist_results = evaluate_model_on_dist_loaders(dist_model)
    utils.print_dict_of_dicts(dist_results)
    print("")

    if args.max_lambda_ < 1e-3 or args.dont_do_critic:
        final_results = dist_results
        final_saver = dist_saver
    else:
        # Fix representations and train pred and aux model
        print("=====================================================")
        print("========= TRAINING ONLY REGULARIZER NOW =============")
        print("========= TRAINING ONLY REGULARIZER NOW =============")
        print("========= TRAINING ONLY REGULARIZER NOW =============")
        print("=====================================================")

        # this will take loaded model and fit
        _, final_results = training_functions.fit_model(
            args_bunch=pred_model_bunch,
            epochs=pred_model_bunch.dist_epochs*2,
            loaders=dist_loaders,
            objective='critic_only',
            writer=SummaryWriter(logdir=FINALSAVEDIR),
            saver=final_saver,
            loaded_model=dist_model
        )

    del dist_model

    # ------------
    # conluding, printing, and saving
    # ------------

    if not args.dont_save_final_results:
        results_pt_filename = "{}/results_{}{}.pt".format(_SAVEFOLDER, args.prefix, "" if args.add_pred_suffix is None else args.add_pred_suffix)
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
