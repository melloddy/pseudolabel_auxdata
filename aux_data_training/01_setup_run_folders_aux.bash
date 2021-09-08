#!/bin/bash

mkdir -p ppv_npv_scan

python prepare_HP_scan_aux.py

for file in ppv_npv_scan/Run_*/
do
	cp submit_aux.sh $file
done

