# hyperparams to try: num_phi_steps, learning rate, different learning rates,
MODIFIER="$1"
GPUTYPE="$2"
WORKERS=2
BS=300
DIST_EPOCHS=5
NR_EPOCHS=10
NUM_FOLDS=5
AMD_PYTHON_STRING="/scratch/apm470/py38_amd/bin/python"
NVIDIA_PYTHON_STRING="/scratch/apm470/py38_nvidia/bin/python"
DATASET="waterbirds"
MODEL="resnet_color"
IMGSIZE=224

GPUSTRING=":mi50"
PYTHON_STRING=$AMD_PYTHON_STRING
if [[ "$GPUTYPE" == "nvidia" ]]; then
    echo "nvidia"
    GPUSTRING=""
    PYTHON_STRING=$NVIDIA_PYTHON_STRING
fi

for method in 'weight' # 'upsample'
do
    for BORDER in 7
    do
        for RHO_TEST in 9
        do
            # if [[ $BORDER != 6 || $RHO_TEST != 9 ]]; then
                for SEED in 1000 2000 3000 4000 5000 6000 7000 8000 9000 500
                    do

                    PREFIX="RWNURD_${method}_${DATASET}_BS${BS}_seed${SEED}_RHOTEST0${RHO_TEST}_BORDER${BORDER}_allwd001_glr0005_FIXEDSPLIT_FINAL"
                    ARGS_STRING="--prefix=$PREFIX --dataset=$DATASET --workers=0 --dist_epochs=$DIST_EPOCHS --nr_epochs=$NR_EPOCHS --nr_batch_size=$BS --dist_batch_size=$BS --seed=$SEED --rho=0.9 --img_side=$IMGSIZE --label_balance_method=downsample --rho_test=0.${RHO_TEST} --border=$BORDER --pred_model_type=$MODEL --weight_model_type=$MODEL --num_folds=$NUM_FOLDS --theta_lr=0.001 --gamma_lr=0.0005 --phi_lr=0.001 --nr_lr=0.001 --dist_decay=0.01 --phi_decay=0.01 --nr_decay=0.01 --nr_strategy=$method --debug=2"

                    # WEIGHT MODEL COMMAND
                    COMMAND_STRING="""
                    
# eval '$PYTHON_STRING nurd_reweighting.py $ARGS_STRING --nr_only --debug=10 >  SBATCH_LOGS/weights_$PREFIX.log'


# reportable
# eval '$PYTHON_STRING nurd_reweighting.py $ARGS_STRING --add_pred_suffix=_LAM1_FRAC2_RR1 --load_weights --lambda_=1 --max_lambda_=1 --frac_phi_steps=2 --randomrestart=1 > SBATCH_LOGS/${PREFIX}_LAM1_FRAC2_RR1.log'

# EQUALIZED gold standard
# eval '$PYTHON_STRING nurd_reweighting.py $ARGS_STRING --add_pred_suffix=_EQUALIZED_ERM --rho=0.5 --rho_test=0.9 --exact --coeff=0.5 --lambda_=0 --max_lambda_=0 --frac_phi_steps=0 --randomrestart=-1 > SBATCH_LOGS/${PREFIX}_EQUALIZED_ERM.log'


# eval '$PYTHON_STRING nurd_reweighting.py $ARGS_STRING --add_pred_suffix=_LAM1_FRAC1_RR1 --load_weights --lambda_=1 --max_lambda_=1 --frac_phi_steps=1 --randomrestart=1 > SBATCH_LOGS/${PREFIX}_LAM1_FRAC1_RR1.log'

# appendix reportable

# eval '$PYTHON_STRING nurd_reweighting.py $ARGS_STRING --add_pred_suffix=_LAM5_FRAC2_RR1 --load_weights --lambda_=5 --max_lambda_=5 --frac_phi_steps=2 --randomrestart=1 > SBATCH_LOGS/${PREFIX}_LAM5_FRAC2_RR1.log'

# ERM

eval '$PYTHON_STRING nurd_reweighting.py $ARGS_STRING --add_pred_suffix=_ERM --exact --coeff=0.5 --lambda_=0 --max_lambda_=0 --frac_phi_steps=0 --randomrestart=-1 > SBATCH_LOGS/${PREFIX}_ERM.log'
                    """

                    printf "\n"
                    printf "\n"
                    printf "\n"
                    echo "$COMMAND_STRING"

                    if [[ "$MODIFIER" == "execute" ]]; then
                        eval "$COMMAND_STRING"
                    fi

                    if [[ "$MODIFIER" == "submit" ]]; then
                        SBATCH_STRING="#!/bin/bash\n#SBATCH --job-name=$PREFIX\n#SBATCH --open-mode=truncate\n#SBATCH --output=./SBATCH_LOGS/$PREFIX.out\n#SBATCH --error=./SBATCH_LOGS/$PREFIX.err\n#SBATCH --export=ALL\n#SBATCH --time=00:10:00\n#SBATCH --gres=gpu${GPUSTRING}:1\n#SBATCH --mem=30G\n#SBATCH -c $WORKERS


    source /home/apm470/.bashrc
    cd /scratch/apm470/projects/nuisance-orthogonal-prediction/code/nurd/

    eval $COMMAND_STRING

    "
    


                        printf "$SBATCH_STRING "
                        printf "$SBATCH_STRING " > RUNNER.sh
                        eval "sbatch RUNNER.sh"
                    fi
                done
            # fi
        done
    done
done