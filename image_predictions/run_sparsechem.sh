#!/bin/bash

predict=$1

mkdir -p cls

{
tstart=`date +%s.%N`
date1=`date`
echo $date1

python $predict \
  --x cls_T11_x_features.npz \
  --y y_sparse_step1_main_tasks_fold2.npy \
  --conf image_model.json \
  --model image_model.pt \
  --outprefix "cls/pred_images_fold2"

tend=`date +%s.%N`
date2=`date`

echo $date2

awk -v tstart=$tstart -v tend=$tend 'BEGIN{time=tend-tstart; printf "TIME [s]: %12.2f\n", time}'
} > out1.dat 2> out2.dat

