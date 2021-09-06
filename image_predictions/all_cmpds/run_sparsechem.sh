#!/bin/bash

predict=$1

mkdir -p preds

{
tstart=`date +%s.%N`
date1=`date`
echo $date1

python $predict \
  --x x_sparse_step2_inference_allcmpds.npz \
  --y y_sparse_step2_inference_allcmpds.npy \
  --conf image_model.json \
  --model image_model.pt \
  --outprefix "preds/pred_cpmodel_step2_inference_allcmpds"

tend=`date +%s.%N`
date2=`date`

echo $date2

awk -v tstart=$tstart -v tend=$tend 'BEGIN{time=tend-tstart; printf "TIME [s]: %12.2f\n", time}'
} > out1.dat 2> out2.dat

