#### Step 0: Prepare Input

Prepare the following files: \
    - T2 that contains all compounds with images (make sure that the used *input_compound_id* match their analogs in your MELLODDY data when they exist). it will later be referred to as T2_images.csv \
    - A file with columns corresponding to image features (standardized) and indexed with the input_compound_id. The file preperation manual is available on box (https://app.box.com/file/852061069648?s=xc2iqr0nylz0p73tj1vfapr2apuuxyx5). In the scripts it's referred to as T_image_features_std.csv

#### Step 1: Image data prepration and image model training

1. Run scripts in 01_datapreparation/ in the order of their numbering:
    - 01_T_generation_images.sh will generate T0, T1, T2 files for image compounds that exist also in Melloddy data with their corresponding tasks.
    - 02_preprocessing_synchronized_thresholds.ipynb will modify T0-2 files to align thresholds with your original melloddy tuner run.
    - 03_run_melloddy_tuner.sh executes melloddy tuner on image only data, using updated T0-2.
    - 04_generating_x_image_features.ipynb creates an analog of X matrix with image features instead of ECFP.  
2. Run scirpts in  02_image_model_training/ :
    - Update submit.sh file in 02_image_model_training/
    - Execute 01_setup_run_folders.bash and 02_submit_all.bash. This will initiate HP scan.
3. Identify the best model using scripts_and_notebooks/step1_3_HP_selection.ipynb - take the last one outputted by the notebook

#### Step 2:

1. Run scripts_and_notebooks/step2_1_ysparse_generation_main_quality_tasks_fold2.ipynb
2. Use the best model identified in 1.3 to execute 03_image_predictions/01_main_tasks_fold2.
    - you need to edit paths in 03_image_predictions/01_main_tasks_fold2/01_link_files.bash and image_predictions/main_tasks_fold2/02_submit_predict.sh
3. Run scripts_and_notebooks/step2_2_CPfitting.ipynb, step2_3_taskstats.ipynb and step2_4_ysparse_inference.ipynb
4. Execute scripts in 03_image_predictions/02_all_cmpds
    - you might need to edit paths in 03_image_predictions/02_all_cmpds/01_link_files.bash and 03_image_predictions/02_all_cmpds/02_submit_predict.sh
5. Run scripts_and_notebooks/step2_5_CPapplication_auxdata.ipynb
    - it refers to T2_images.csv, this file is described in the beginning of the readme. It contain all image compounds, not only ones in MELLODDY
6. Run scripts_and_notebooks/step2_6_Tfilegeneration.ipynb
7. Run scripts_and_notebooks/step2_7_labels_to_auxtasks.ipynb
    - tuner_output_baseline there refers to results of melloddy tuner without images
8. Run scripts_and_notebooks/step2_8_concat_label_imputation.ipynb

#### Step 3:

1. Execute scripts in 04_aux_data_preparation/
    - will take as much time as a melloddy tuner run on your original data (or more :))
2. Run scripts_and_notebooks/step3_1_confidence_selection.ipynb
3. Execute scripts in 05_aux_data_training/
    - you need to edit paths in submit_aux.sh
    - in 00_best_hyperparameters.dat change model HPs to the optimal ones for your MELLODDY dataset
4. Run scripts_and_notebooks/step3_2_y_sparse_generation_inference.ipynb
5. Execute scripts in 06_aux_data_predictions/
    - edit paths/envs in submit_baseline.sh and submit_aux.sh
6. Run scripts_and_notebooks/step3_3_evaluation.ipynb

#### Additional steps for trying different aux weights:

1. Execute aux_data_preparation/02_generate_cls_weights.sh - this will generate different weight files
2. Modify the best HPs in 05_aux_data_training/00_best_hyperparameters.dat to be the same as in 05_aux_data_training/03_run_sparsechem_baseline.sh
3. Modify 05_aux_data_training/submit.sh analogously to submit_aux.sh
4. Run 05_aux_data_training/01_setup_run_folders_aux.bash
5. Run 05_aux_data_training/02_submit_all_aux.bash
    - if you want to skip runs that were already done, modify the corresponding foldernames
