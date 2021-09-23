#!/bin/bash

predict=$1
data_path=$2


model_path=../05_aux_data_training/baseline/models

mkdir -p preds_baseline


{
tstart=`date +%s.%N`
date1=`date`
echo $date1

python $predict \
  --x $data_path/cls/cls_T11_x.npz \
  --y $data_path/cls/fold_0/y_sparse_main_tasks_fold0_baseline.npz \
  --conf $model_path/classification_baseline.json \
  --model $model_path/classification_baseline.pt \
  --outprefix "preds_baseline/pred_aux_fold0"

tend=`date +%s.%N`
date2=`date`

echo $date2

awk -v tstart=$tstart -v tend=$tend 'BEGIN{time=tend-tstart; printf "TIME [s]: %12.2f\n", time}'
} > out1.dat 2> out2.dat

