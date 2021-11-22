import os
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from pseudolabel.constants import IMAGE_MODEL_NAME
from pseudolabel.errors import PredictOptError


def create_x_ysparse_all_images(
    tuner_output_image: str,
    t_images_features_path: str,
    analysis_folder: str,
    intermediate_files_folder: str,
):
    t10_path = os.path.join(tuner_output_image, "matrices/cls/cls_T10_y.npz")
    t10 = load_npz(t10_path)

    x_arr = pd.read_csv(t_images_features_path, index_col=0)

    cp_results_path = os.path.join(analysis_folder, "cp/summary_eps_0.05.csv")

    cp_res = pd.read_csv(cp_results_path)

    # efficiencies of 0 will be filtered out
    # low data volume regime
    # reduced hope for obtaining both labels from the CP

    col_idxs = cp_res.query("validity_0 >= 0").query("validity_1 >= 0")["index"]

    num_rows, num_cols = x_arr.shape[0], t10.shape[1]

    data = np.ones(num_rows * col_idxs.shape[0])
    rows = np.repeat(np.arange(num_rows), col_idxs.shape[0])
    cols = np.tile(col_idxs.values, num_rows)

    Y_to_pred = csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))

    save_npz(
        file=os.path.join(
            intermediate_files_folder, "y_sparse_step2_inference_allcmpds.npz"
        ),
        matrix=csr_matrix(Y_to_pred),
    )

    save_npz(
        file=os.path.join(
            intermediate_files_folder, "x_sparse_step2_inference_allcmpds.npz"
        ),
        matrix=csr_matrix(x_arr),
    )


def run_sparsechem_predict(
    sparsechem_predictor_path: str,
    best_model: str,
    intermediate_files_folder: str,
    dataloader_num_workers: int,
    torch_device: str,
    logs_dir: Optional[str] = None,
):
    x_path = os.path.join(
        intermediate_files_folder, "x_sparse_step2_inference_allcmpds.npz"
    )
    y_path = os.path.join(
        intermediate_files_folder, "y_sparse_step2_inference_allcmpds.npz"
    )
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)

    proc = subprocess.run(
        [
            "python",
            sparsechem_predictor_path,
            "--x",
            x_path,
            "--y",
            y_path,
            "--conf",
            os.path.join(best_model, f"{IMAGE_MODEL_NAME}.json"),
            "--model",
            os.path.join(best_model, f"{IMAGE_MODEL_NAME}.pt"),
            "--outprefix",
            os.path.join(
                intermediate_files_folder, "pred_cpmodel_step2_inference_allcmpds"
            ),
            "--dev",
            torch_device,
            "--num_workers",
            str(dataloader_num_workers),
        ],
        stdout=open(
            os.path.join(logs_dir, "pred_cpmodel_step2_inference_allcmpds_log.txt"), "w"
        )
        if logs_dir
        else subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        raise PredictOptError(f"Predict all images failed: \n {proc.stderr.decode()}")
