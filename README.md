Step 0: data preparation

1. Prepare the following files::

- T0, T1, T2 that contian the subset of your MELLODDY data that correspond to compounds with images (make sure to not change input_compound_id and input_assay_id, they should match their analogs in your MELLODDY data)
- a file with columns corresponding to image features (standardized) and input_compound_id. In the scripts it's referred to as T_image_features_std.csv


2.Run scripts in datapreparation/ in the order of their numbering:

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
