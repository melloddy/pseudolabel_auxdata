#!/bin/bash

#SBATCH --job-name=Train_classification_no_aux
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu
#SBATCH --mem-per-cpu=4G

# Activate conda environment containing SparseChem


# Full path to SparseChem train.py (Version: 0.8.2), e.g.: /home/user/sparsechem/examples/chembl/train.py
predict_path=/home/rama.jabal/Melloddy/sparsechem/examples/chembl/predict.py

# Full path to matrices, e.g.: /home/user/data_prep/out/run/matrices/
data_path=/home/rama.jabal/Melloddy/aux_data/pseudolabel_auxdata/04_aux_data_preperation/baseline_plus_aux_data/matrices
if type srun > /dev/null 2>&1; then
    srun ./run_sparsechem_aux.sh $predict_path $data_path
else
    sh ./run_sparsechem_aux.sh $predict_path $data_path
fi
