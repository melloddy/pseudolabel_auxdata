#!/bin/bash

path_matrices='../../scripts_and_notebooks/files'

path_model="../../image_model_training/HP_scan_image_model/Run_231_epoch_lr_step_20_10_drop_0.2_size_125125125/models/"

# 01: link data
ln -s $path_matrices/x_sparse_step2_inference_allcmpds.npz
ln -s $path_matrices/y_sparse_step2_inference_allcmpds.npy

# 02: link model

ln -s $path_model/image_model.json
ln -s $path_model/image_model.pt

