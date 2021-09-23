#!/bin/bash


params_dir='../../data/Input' # path to parameters.json file: ./example_parameters.json

path_T_files='./images_files/input'

outdir='./images_files/output_files'


tunercli prepare_4_training \
--structure_file $path_T_files/T2_synchedthreshold.csv \
--activity_file $path_T_files/T1_synchedthreshold.csv \
--weight_table $path_T_files/T0_synchedthreshold.csv \
--config_file $params_dir/parameters.json \
--key_file $params_dir/key.json \
--output_dir $outdir \
--run_name image_model \
--tag cls \
--folding_method scaffold \
--number_cpu 8 \
--ref_hash $params_dir/ref_hash.json \
