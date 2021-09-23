#!/bin/bash



path_T_files='../scripts_and_notebooks/files/baseline_plus_aux_data/'
params_dir=$(cat ./00_parameter_path.dat)


outdir=.
    
tunercli prepare_4_training \
--structure_file $path_T_files/T2_image_pseudolabels_plus_baseline.csv \
--activity_file $path_T_files/T1_image_pseudolabels_plus_baseline.csv \
--weight_table $path_T_files/T0_image_pseudolabels_plus_baseline.csv \
--config_file $params_dir/parameters.json \
--key_file $params_dir/key.json \
--output_dir $outdir \
--run_name baseline_plus_aux_data \
--tag cls \
--folding_method scaffold \
--number_cpu 8 \
--ref_hash $params_dir/ref_hash.json 
    


