# hyperparams to try: num_phi_steps, learning rate, different learning rates,
MODIFIER="$1"
GPUTYPE="$2"
WORKERS=4
BS=1000
DIST_EPOCHS=100
NR_EPOCHS=150
NUM_FOLDS=5
AMD_PYTHON_STRING="/scratch/apm470/py38_amd/bin/python"
NVIDIA_PYTHON_STRING="/scratch/apm470/py38_nvidia/bin/python"
DATASET="joint"
MODEL="small"
IMGSIZE=32
# DATABALANCETYPE="downsample"

GPUSTRING=":mi50"
PYTHON_STRING=$AMD_PYTHON_STRING
if [[ "$GPUTYPE" == "nvidia" ]]; then
    echo "nvidia"
    GPUSTRING=""
    PYTHON_STRING=$NVIDIA_PYTHON_STRING
fi

for DATABALANCETYPE in "downsample"
do
    for method in 'weight'
    do
        for BORDER in 6
        do
            for RHO_TEST in 9
            do
                # if [[ $BORDER != 6 || $RHO_TEST != 9 ]]; then
                    for SEED in 1000 2000 3000 4000 5000 6000 7000 8000 9000 500
                        do

                        PREFIX="GENNURD_${method}_${DATASET}_BAL${DATABALANCETYPE}_BS${BS}_seed${SEED}_RHOTEST0${RHO_TEST}_BORDER${BORDER}_noBN_LONGGEN"
                        ARGS_STRING="--prefix=$PREFIX --dataset=$DATASET --workers=0 --dist_epochs=$DIST_EPOCHS --nr_epochs=$NR_EPOCHS --nr_batch_size=$BS --dist_batch_size=$BS --seed=$SEED --rho=0.9 --img_side=$IMGSIZE --label_balance_method=$DATABALANCETYPE --rho_test=0.${RHO_TEST} --border=$BORDER --pred_model_type=$MODEL --weight_model_type=$MODEL --num_folds=$NUM_FOLDS --theta_lr=0.001 --gamma_lr=0.001 --phi_lr=0.001 --nr_lr=0.001 --dist_decay=0.0 --phi_decay=0.0 --nr_decay=0.01 --nr_strategy=$method --exact --coeff=0.5 --debug=2 --train_filename=/scratch/apm470/projects/generative_models/SAVED_DATA/nurd_chestxray_long_gen_${SEED}_generated_data.pt"
                        # nurd_gen_${SEED}_generated_data.pt"

                    COMMAND_STRING="""

# 1 epoch per pred update, with randomrestart=-1
eval '$PYTHON_STRING nurd_reweighting.py $ARGS_STRING --add_pred_suffix=_DEFLR_distWD0_LAM1_FRAC1_RR1_LONG --dist_decay=0.0 --lambda_=1 --max_lambda_=1 --frac_phi_steps=1 --randomrestart=1 > SBATCH_LOGS/${PREFIX}_DEFLR_distWD0_LAM1_FRAC1_RR1_LONG.log'

# 2 epoch per pred update, with randomrestart=1
eval '$PYTHON_STRING nurd_reweighting.py $ARGS_STRING --add_pred_suffix=_DEFLR_distWD0_LAM1_FRAC2_RR1_LONG --dist_decay=0.0 --lambda_=1 --max_lambda_=1 --frac_phi_steps=2 --randomrestart=1 > SBATCH_LOGS/${PREFIX}_DEFLR_distWD0_LAM1_FRAC2_RR1_LONG.log'


# 1 epoch per pred update, with randomrestart=-1
# eval '$PYTHON_STRING nurd_reweighting.py $ARGS_STRING --add_pred_suffix=_DEFLR_distWD10_LAM1_FRAC1_RR1_LONG --dist_decay=1.0 --lambda_=1 --max_lambda_=1 --frac_phi_steps=1 --randomrestart=1 > SBATCH_LOGS/${PREFIX}_DEFLR_distWD01_LAM1_FRAC1_RR1_LONG.log'

# 2 epoch per pred update, with randomrestart=1
# eval '$PYTHON_STRING nurd_reweighting.py $ARGS_STRING --add_pred_suffix=_DEFLR_distWD10_LAM1_FRAC2_RR1_LONG --dist_decay=1.0 --lambda_=1 --max_lambda_=1 --frac_phi_steps=2 --randomrestart=1 > SBATCH_LOGS/${PREFIX}_DEFLR_distWD01_LAM1_FRAC2_RR1_LONG.log'
                    """

                        printf "\n"
                        if [[ "$MODIFIER" == "print" ]]; then
                            echo $COMMAND_STRING
                        fi

                        if [[ "$MODIFIER" == "execute" ]]; then
                            eval "$COMMAND_STRING"
                        fi

                        if [[ "$MODIFIER" == "submit" ]]; then
                            SBATCH_STRING="#!/bin/bash\n#SBATCH --job-name=$PREFIX\n#SBATCH --open-mode=truncate\n#SBATCH --output=./SBATCH_LOGS/$PREFIX.out\n#SBATCH --error=./SBATCH_LOGS/$PREFIX.err\n#SBATCH --export=ALL\n#SBATCH --time=00:30:00\n#SBATCH --gres=gpu${GPUSTRING}:1\n#SBATCH --mem=20G\n#SBATCH -c $WORKERS


source /home/apm470/.bashrc
cd /scratch/apm470/projects/nuisance-orthogonal-prediction/code/nurd/

$COMMAND_STRING

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
done