#!/bin/bash

mkdir -p ppv_npv_preds

python prepare_preds_aux.py

for file in ppv_npv_preds/Preds_*/
do
	cp submit_aux.sh $file
done

