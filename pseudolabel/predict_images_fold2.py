import glob
import json
import logging
import os
import os.path
import subprocess
import types
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, load_npz
from tqdm import tqdm

from pseudolabel.constants import IMAGE_MODEL_NAME
from pseudolabel.errors import PredictOptError

LOGGER = logging.getLogger(__name__)


def load_results(filename: str, two_heads: bool = False):  # noqa
    """Loads conf and results from a file
    Args:
        filename    name of the json/npy file
        two_heads   set up class_output_size if missing
    """
    # TODO Simplify this
    if filename.endswith(".npy"):
        return np.load(filename, allow_pickle=True).item()

    with open(filename, "r") as f:
        data = json.load(f)

    for key in ["model_type"]:
        if key not in data["conf"]:
            data["conf"][key] = None
    if two_heads and ("class_output_size" not in data["conf"]):
        data["conf"]["class_output_size"] = data["conf"]["output_size"]
        data["conf"]["regr_output_size"] = 0

    data["conf"] = types.SimpleNamespace(**data["conf"])

    if "results" in data:
        for key in data["results"]:
            data["results"][key] = pd.read_json(data["results"][key])

    if "results_agg" in data:
        for key in data["results_agg"]:
            data["results_agg"][key] = pd.read_json(
                data["results_agg"][key], typ="series"
            )

    for key in ["training", "validation"]:
        if key not in data:
            continue
        for dfkey in ["classification", "regression"]:
            data[key][dfkey] = pd.read_json(data[key][dfkey])
        for skey in ["classification_agg", "regression_agg"]:
            data[key][skey] = pd.read_json(data[key][skey], typ="series")

    return data


def find_best_model(hyperopt_folder: str, eval_metric: str = "roc_auc_score") -> str:
    avg_perf = 0
    best_model = None
    models = glob.glob(os.path.join(hyperopt_folder, "*", "*.json"))

    if not models:
        raise FileNotFoundError(f"Can't find any model in {hyperopt_folder}")

    for file in tqdm(models):
        res = load_results(file)
        new_avg_perf = np.mean(res["validation"]["classification_agg"][eval_metric])
        if avg_perf < new_avg_perf:
            avg_perf = new_avg_perf
            best_model = os.path.dirname(file)

    if best_model is None:
        raise ValueError(f"No valid model found in {hyperopt_folder}")

    LOGGER.debug(f"Found best model: {best_model}")

    return best_model


def create_ysparse_fold2(tuner_output_images: str, intermediate_files_folder: str):
    folds_path = os.path.join(
        tuner_output_images, "matrices", "cls", "cls_T11_fold_vector.npy"
    )
    folds = np.load(folds_path)
    # TODO check if npy is never used
    t10_path = os.path.join(tuner_output_images, "matrices", "cls", "cls_T10_y.npz")
    t10 = load_npz(t10_path).tocsr()

    out = lil_matrix(t10)

    fold = 2
    out[folds != fold, :] = 0

    t8c_path = os.path.join(
        tuner_output_images, "results_tmp", "classification", "T8c.csv"
    )
    t8c = pd.read_csv(t8c_path)

    cctis = np.array(
        t8c.query("is_auxiliary == False").query("aggregation_weight > 0")[
            "cont_classification_task_id"
        ],
        dtype=int,
    )

    os.makedirs(intermediate_files_folder, exist_ok=True)

    np.save(os.path.join(intermediate_files_folder, "main_tasks_with_image.npy"), cctis)

    # mind the ~
    mask = ~np.isin(
        np.array(range(int(t8c["cont_classification_task_id"].max() + 1))), cctis
    )

    # ignore all non-main tasks
    out[:, mask] = 0

    filename = os.path.join(
        intermediate_files_folder, "y_sparse_step1_main_tasks_fold2.npy"
    )
    np.save(filename, csr_matrix(out))


def run_sparsechem_predict(
    sparsechem_predictor_path: str,
    tuner_output_dir: str,
    best_model: str,
    intermediate_files_folder: str,
    dataloader_num_workers: int,
    torch_device: str,
    logs_dir: Optional[str] = None,
):
    x_path = os.path.join(tuner_output_dir, "matrices", "cls", "cls_T11_x_features.npz")
    y_path = os.path.join(
        intermediate_files_folder, "y_sparse_step1_main_tasks_fold2.npy"
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
            os.path.join(intermediate_files_folder, "pred_images_fold2"),
            "--dev",
            torch_device,
            "--num_workers",
            str(dataloader_num_workers),
        ],
        stdout=open(os.path.join(logs_dir, "predict_images_fold2_log.txt"), "w")
        if logs_dir
        else subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        raise PredictOptError(f"Predict images fold failed: \n {proc.stderr.decode()}")
