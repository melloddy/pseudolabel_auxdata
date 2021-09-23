#!/bin/bash

T_melloddy='/home/rama.jabal/Melloddy/aux_data/data/Mellody_tuner/all/input_all'
T_all_images='/home/rama.jabal/Melloddy/aux_data/ci8b00670_si_002/Label_Matrix/T_files/images'
output_dir='./images_files/input'

mkdir -p $output_dir 

python T_generation.py $T_melloddy $T_all_images $output_dir