#!/bin/bash

#SBATCH --job-name=Predict_cls
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu
#SBATCH --mem-per-cpu=4G

# Activate conda environment containing SparseChem

predict_path="/home/rama.jabal/Melloddy/sparsechem/examples/chembl/predict.py"

if type srun > /dev/null 2>&1; then
    srun ./run_sparsechem.sh $predict_path
else
    sh ./run_sparsechem.sh $predict_path
fi
