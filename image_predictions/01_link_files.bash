#!/bin/bash

path_matrices='../scripts_and_notebooks/files'
tuner_output_images='../datapreperation/images_files/output_files/image_model'

path_model="../image_model_training/HP_scan_image_model/Run_231_epoch_lr_step_20_10_drop_0.2_size_125125125/models/"

# 01: link data
ln -s $path_matrices/y_sparse_step1_main_tasks_fold2.npy
ln -s $tuner_output_images/matrices/cls/cls_T11_x_features.npz

# 02: link model

ln -s $path_model/image_model.json
ln -s $path_model/image_model.pt

