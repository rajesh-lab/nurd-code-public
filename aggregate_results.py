#aggregate_results.py
from scipy import stats
import numpy as np
import glob
import sys

import numpy as np
import torch

key_list = ["train___pred_loss", "train___aux_loss", "train___kl_loss", "train___acc", "bal_val___pred_loss", "bal_val___aux_loss", "bal_val___kl_loss", "bal_val___acc", "unbal_val___pred_loss", "unbal_val___aux_loss", "unbal_val___kl_loss", "unbal_val___acc", "test___pred_loss", "test___aux_loss", "test___kl_loss", "test___acc"]

# key_list = ["train___pred_loss", "train___aux_loss", "train___kl_loss", "train___acc", "bal_val___pred_loss", "bal_val___aux_loss", "bal_val___kl_loss", "bal_val___acc", "test___pred_loss", "test___aux_loss", "test___kl_loss", "test___acc"]

print_list = ["bal_val___pred_loss", "bal_val___aux_loss", "bal_val___kl_loss", "bal_val___acc", "test___acc"]

key_jointindep_list = ["train___pred_loss", "train___critic_loss", "train___info_loss", "train___acc", "train___critic_acc", "train___critic_acc_debug", "bal_val___pred_loss", "bal_val___critic_loss", "bal_val___info_loss", "bal_val___acc", "bal_val___critic_acc", "bal_val___critic_acc_debug", "unbal_val___pred_loss", "unbal_val___critic_loss", "unbal_val___info_loss", "unbal_val___acc", "unbal_val___critic_acc", "unbal_val___critic_acc_debug", "test___pred_loss", "test___critic_loss", "test___info_loss", "test___acc", "test___critic_acc", "test___critic_acc_debug"]

# key_jointindep_list = ["train___pred_loss", "train___critic_loss", "train___info_loss", "train___acc", "train___critic_acc", "train___critic_acc_debug", "bal_val___pred_loss", "bal_val___critic_loss", "bal_val___info_loss", "bal_val___acc", "bal_val___critic_acc", "bal_val___critic_acc_debug", "test___pred_loss", "test___critic_loss", "test___info_loss", "test___acc", "test___critic_acc", "test___critic_acc_debug"]


key_jointindep_list += ["rev_val___pred_loss", "rev_val___critic_loss", "rev_val___info_loss", "rev_val___acc", "rev_val___critic_acc", "rev_val___critic_acc_debug"]
print_jointindep_list = ["bal_val___pred_loss", "bal_val___info_loss", "bal_val___acc", "test___acc"]

key_jointindep_list += ["val___pred_loss", "val___critic_loss", "val___info_loss", "val___acc", "val___critic_acc", "val___critic_acc_debug"]
print_jointindep_list = ["val___pred_loss", "val___info_loss", "val___acc", "test___acc"]

key_jointindep_list += ['eval___acc', 'eval___pred_loss']
print_jointindep_list += ['eval___acc', 'eval___pred_loss']



filters = [] #lambda x : x["bal_val__acc"] > 0.51]

def loss_func(dictionary, lambda_):
    if "bal_val___kl_loss" in dictionary.keys():
        return dictionary["bal_val___pred_loss"] + lambda_*dictionary["bal_val___kl_loss"]
    elif "val___info_loss" in dictionary.keys():
        return dictionary["val___pred_loss"] + lambda_*dictionary["val___info_loss"]
    else:
        return dictionary["bal_val___pred_loss"] + lambda_*dictionary["bal_val___info_loss"]

functions = {"bal_val__full_loss" : loss_func}

def parse_results(glob_string, lambda_, print_list=print_list, filters=filters, functions=functions, avg_only=False, no_print=False):

    result_strings = glob.glob(glob_string)
    print("FOUND {} RESULTS".format(len(result_strings)))
    # if "INDEP" in glob_string:
    print_list = print_jointindep_list
    key_list = key_jointindep_list
    
    if not no_print:
        print("USED" , glob_string)
        # print(result_strings)
    print("GOT THIS MANY", len(result_strings))

    print_list = print_list + list(functions.keys())
    
    try:
        result_strings = sorted(result_strings, key=lambda x: int(x[-4]))
    except:
        if not no_print:
            print("No noise ordering.")

    agg_list = {print_name : [] for print_name in key_list + list(functions.keys())}

    if not no_print:
        print(" | ".join(["{:20s}".format(key) for key in print_list]))
    
    for result_string in result_strings:
        
        result = torch.load(result_string)
        
        try:
            if len(result["weight_model"]) > 0:
                weight_val_loss_dict_list = result['weight_model']
                # print(weight_val_loss_dict_list)
            else:
                weight_val_loss_dict_list = None
        except:
            weight_val_loss_dict_list = None

            
        pred_result_dict = result['pred_models']
        
        for key, func in functions.items():
            pred_result_dict[key] = func(pred_result_dict, lambda_)

    
        if weight_val_loss_dict_list is not None:
            if "weight_model_acc" in agg_list.keys():
                agg_list["weight_model_acc"] += [weight_val_loss_dict_list[0]["s_acc"]]
                agg_list["weight_model_loss"] += [weight_val_loss_dict_list[0]["s_loss"]]
            else:
                agg_list["weight_model_acc"] = [weight_val_loss_dict_list[0]["s_acc"]]
                agg_list["weight_model_loss"] = [weight_val_loss_dict_list[0]["s_loss"]]
            pred_result_dict["weight_model_acc"] = weight_val_loss_dict_list[0]["s_acc"]
            pred_result_dict["weight_model_loss"] = weight_val_loss_dict_list[0]["s_loss"]
            if "weight_model_acc" not in print_list:
                print_list.append("weight_model_acc")

        # filter out
        skip_flag = False
        for filter in filters:
            if filter(pred_result_dict):
                skip_flag = True
                break
        if skip_flag:
            continue

        # assert False kl_flag
        # print(pred_result_dict)


        for (key, metric) in pred_result_dict.items():
            if key in key_list + list(functions.keys()):
                agg_list[key].append(metric)
        
        if not avg_only and not no_print:
            print(" | ".join(["{:20.4f}".format(pred_result_dict[key]) for key in print_list]))

    consolidated_list = {}

    for (key, val_list) in agg_list.items():
        if len(val_list) < 1:
            continue
        if not no_print:
            print(" {:25s} {:.3f} \pm {:.3f}".format(key, np.mean(val_list), np.std(val_list)/np.sqrt(len(val_list))))
        
        consolidated_list[key]={
            "mean" :  np.mean(val_list),
            "sem" : np.std(val_list)/np.sqrt(len(val_list)),
            "list" : val_list
        }

    for (key, val_list) in agg_list.items():
        if len(val_list) < 1:
            continue
        if "test" in key and "loss" not in key and not no_print:
            print(" MIN/MAX {:20s}  = {:.2f}/{:.2f}".format(key, np.min(val_list), np.max(val_list)))
    
    return agg_list, consolidated_list

if __name__ == '__main__':
    glob_string = sys.argv[1]
    parse_results(glob_string)