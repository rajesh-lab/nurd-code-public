# nurd-code-public

Companion code for the paper [Out-of-distribution Generalization in the Presence of Nuisance-Induced Spurious Correlations](https://openreview.net/forum?id=12RoR2o32T).

Email <aahlad@nyu.edu> to get the datasets used in the paper. You can also generate them using `save_processed_dataset_as_pt.py` which takes data and stores them as pytorch tensors to make training very fast. If you want to use the raw data, please make a dataset with the functionality of the class `XYZ_DatasetWithIndices` in the `dataloaders.py` file.

 > example run for the chest X-ray experiment with default distillation

 ```
 time python  nurd_reweighting.py --prefix=DUMMY --dataset=joint --workers=0 --dist_epochs=10 --nr_epochs=15 --nr_batch_size=1000 --dist_batch_size=1000 --seed=500 --rho=0.9 --img_side=32 --label_balance_method=downsample --rho_test=0.9 --border=6 --pred_model_type=small --weight_model_type=small --num_folds=2 --theta_lr=0.001 --gamma_lr=0.001 --phi_lr=0.001 --nr_lr=0.001 --dist_decay=0.0 --phi_decay=0.0 --nr_decay=0.01 --nr_strategy=weight --debug=2
 ```

 > example run for waterbirds with specified distillation

 ```
 python nurd_reweighting.py --prefix=DUMMY --dataset=waterbirds --workers=0 --dist_epochs=1 --nr_epochs=2 --nr_batch_size=300 --dist_batch_size=300 --seed=500 --rho=0.9 --img_side=224 --label_balance_method=downsample --rho_test=0.9 --border=7 --pred_model_type=resnet_color --weight_model_type=resnet_color --num_folds=2 --theta_lr=0.001 --gamma_lr=0.0005 --phi_lr=0.001 --nr_lr=0.001 --dist_decay=0.01 --phi_decay=0.01 --nr_decay=0.01 --nr_strategy=weight --debug=2 --add_pred_suffix=_LAM1_FRAC2_RR1 --load_weights --lambda_=1 --max_lambda_=1 --frac_phi_steps=2 --randomrestart=1
 ```


Use `--nr_only` to learn only the weight model. Weights are saved by default and can be loaded using `--load_weights`.


 # required folders

 Create the following folders in the directory for running the scripts. 

 ```
LOGS/
SAVED_DATA/
SAVED_MODELS/
cub/
```

`cub/` is the directory to put the waterbirds data.
