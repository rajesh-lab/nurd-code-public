# hyperparams to try: num_phi_steps, learning rate, different learning rates,
MODIFIER="$1"
GPUTYPE="$2"
WORKERS=2
BS=300
DIST_EPOCHS=40
NR_EPOCHS=20
NUM_FOLDS=5
AMD_PYTHON_STRING="/scratch/apm470/py38_amd/bin/python"
NVIDIA_PYTHON_STRING="/scratch/apm470/py38_nvidia/bin/python"
DATASET="cmnist"
MODEL="cmnist"
IMGSIZE=28

GPUSTRING=":mi50"
PYTHON_STRING=$AMD_PYTHON_STRING
if [[ "$GPUTYPE" == "nvidia" ]]; then
    echo "nvidia"
    GPUSTRING=""
    PYTHON_STRING=$NVIDIA_PYTHON_STRING
fi

for method in 'weight' # 'upsample'
do
    for BORDER in 6
    do
        for RHO_TEST in 9
        do
            # if [[ $BORDER != 6 || $RHO_TEST != 9 ]]; then
                for SEED in 1000 2000 3000 4000 5000 6000 7000 8000 9000 500
                    do

                    PREFIX="GENNURD_${method}_${DATASET}_BS${BS}_seed${SEED}_RHOTEST0${RHO_TEST}_BORDER${BORDER}_DEF_FINAL"
                    ARGS_STRING="--prefix=$PREFIX --dataset=$DATASET --workers=0 --dist_epochs=$DIST_EPOCHS --nr_epochs=$NR_EPOCHS --nr_batch_size=$BS --dist_batch_size=$BS --seed=$SEED --rho=0.9 --img_side=$IMGSIZE --rho_test=0.${RHO_TEST} --pred_model_type=$MODEL --weight_model_type=$MODEL --num_folds=$NUM_FOLDS --nr_strategy=$method --debug=2 --exact --coeff=0.5"

                    # WEIGHT MODEL COMMAND
                    COMMAND_STRING="""
eval '$PYTHON_STRING evaluate_model.py $ARGS_STRING --add_pred_suffix=_LAM1_FRAC1_NORR --load_final_model --add_final_results_suffix=SAVED --lambda_=1 --max_lambda_=1 --frac_phi_steps=1 --randomrestart=-1'
                    """

                    printf "\n"
                    echo "$COMMAND_STRING"

                    if [[ "$MODIFIER" == "execute" ]]; then
                    	eval "$COMMAND_STRING"
                    fi

                    if [[ "$MODIFIER" == "submit" ]]; then
                        SBATCH_STRING="#!/bin/bash\n#SBATCH --job-name=$PREFIX\n#SBATCH --open-mode=truncate\n#SBATCH --output=./SBATCH_LOGS/$PREFIX.out\n#SBATCH --error=./SBATCH_LOGS/$PREFIX.err\n#SBATCH --export=ALL\n#SBATCH --time=00:30:00\n#SBATCH --gres=gpu${GPUSTRING}:1\n#SBATCH --mem=10G\n#SBATCH -c $WORKERS


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