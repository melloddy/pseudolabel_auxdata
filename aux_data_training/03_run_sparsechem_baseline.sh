#!/bin/bash

train='/home/rama.jabal/Melloddy/sparsechem/examples/chembl/train.py'
data_path='/home/rama.jabal/Melloddy/aux_data/pseudolabel_auxdata/aux_data_preperation/baseline_plus_aux_data/matrices'

# Hyperparameters to be changed - please do not edit the placeholders
hidden_sizes='1200'
dropout=0.8
lr_steps=10
epochs=20

{
tstart=`date +%s.%N`
date1=`date`
echo $date1

python $train \
  --x $data_path/cls/cls_T11_x.npz \
  --y $data_path/cls/confidence_selection/cls_T10_y_baseline.npz \
  --folding $data_path/cls/cls_T11_fold_vector.npy \
  --weights_class $data_path/cls/cls_weights.csv \
  --hidden_sizes $hidden_sizes \
  --last_dropout $dropout \
  --middle_dropout $dropout \
  --last_non_linearity relu \
  --non_linearity relu \
  --input_transform none \
  --lr 0.001 \
  --lr_alpha 0.3 \
  --lr_steps $lr_steps \
  --epochs $epochs \
  --normalize_loss 100_000 \
  --eval_frequency 1 \
  --batch_ratio 0.02 \
  --fold_va 2 \
  --fold_te 0 \
  --verbose 1 \
  --save_model 1 \
  --run_name classification_baseline

tend=`date +%s.%N`
date2=`date`

echo $date2

awk -v tstart=$tstart -v tend=$tend 'BEGIN{time=tend-tstart; printf "TIME [s]: %12.2f\n", time}'
} > out1.dat 2> out2.dat

