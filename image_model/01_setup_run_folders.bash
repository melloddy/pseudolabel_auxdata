#!/bin/bash

mkdir -p HP_scan_image_model

python prepare_HP_scan.py

for file in HP_scan_image_model/Run_*/
do
	cp submit.sh $file
done

