#!/bin/bash

python prepare_HP_scan_aux.py
python prepare_HP_scan_baseline.py

for file in ppv_npv_scan*/Run_*/
do
	cp submit.sh $file
done

for file in baseline_*/
do
	cp submit.sh $file
done
