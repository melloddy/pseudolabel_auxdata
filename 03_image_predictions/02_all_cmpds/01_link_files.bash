#!/bin/bash

path_matrices='../../scripts_and_notebooks/files'

path_model="../../02_image_model_training/HP_scan_image_model/Run_249_epoch_lr_step_20_10_drop_0.1_size_125125125/models/" #path towards the best image-based model

# 01: link data
ln -s $path_matrices/x_sparse_step2_inference_allcmpds.npz
ln -s $path_matrices/y_sparse_step2_inference_allcmpds.npz

# 02: link model

ln -s $path_model/image_model.json
ln -s $path_model/image_model.pt

