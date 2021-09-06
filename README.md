Step 0: data preparation

1. Prepare the following files::
    - T0, T1, T2 that contian the subset of your MELLODDY data that correspond to compounds with images (make sure to not change input_compound_id and input_assay_id, they should match their analogs in your MELLODDY data)
    - a file with columns corresponding to image features (standardized) and input_compound_id. In the scripts it's referred to as T_image_features_std.csv
    - a T2 file with all image compounds, it will later be referred to as T2_images.csv
2. Run scripts in datapreparation/ in the order of their numbering:
    - 01_preprocessing_synchronized_thresholds.ipynb will modify T0-2 files to align thresholds with your original melloddy tuner run
    - 02_run_melloddy_tuner.sh executes melloddy tuner on image only data, using updated T0-2
    - 03_generating_x_image_features.ipynb creates an analog of X matrix with image features instead of ECFP

Step 1: model training using image features

1. Update submit.sh in image_model_training
2. Execute 01_setup_run_folders.bash and 02_submit_all.bash. This will initiate HP scan.
3. Identify the best model using scripts_and_notebooks/step1_3_HP_selection.ipynb - take the first one outputted (CHECK WITH WOUTER) 

Step 2:
1. Run scripts_and_notebooks/step2_1_ysparse_generation_main_quality_tasks_fold2.ipynb
2. Use that model to execute image_predictions/main_tasks_fold2
    - you might need to edit paths in image_predictions/main_tasks_fold2/01_link_files.bash and image_predictions/main_tasks_fold2/02_submit_predict.sh
3. Run scripts_and_notebooks/step2_2_CPfitting.ipynb, step2_3_taskstats.ipynb and step2_4_ysparse_inference.ipynb
4. Execute scripts in image_predictions/all_cmpds
    - you might need to edit paths in image_predictions/all_cmpds/01_link_files.bash and image_predictions/all_cmpds/02_submit_predict.sh
    - you might need to create preds/ folder before executing 02_submit_predict.sh
5. Run scripts_and_notebooks/step2_5_CPapplication_auxdata.ipynb 
    - it refers to T2_images.csv, this file should contain all image compounds, not only ones in MELLODDY
6. Run scripts_and_notebooks/step2_6_Tfilegeneration.ipynb
7. Run scripts_and_notebooks/step2_7_labels_to_auxtasks.ipynb
    - tuner_output_baseline there refers to results of melloddy tuner without images 
8. Run scripts_and_notebooks/step2_8_concat_label_imputation.ipynb
