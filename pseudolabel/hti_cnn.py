import json
import os
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import torch

from pseudolabel.constants import T2_IMAGES, T_IMAGE_FEATURES
from pseudolabel.errors import HTIError


def create_featurizer_config(train_config_file: str):
    train_config = json.load(open(train_config_file))
    train_config.update({"model": train_config["model"] + "Featurizer"})
    train_config["dataset_to_eval"] = "val"
    train_config["dataset"]["val"] = train_config["dataset"]["train"]
    train_config["dataset"].pop("train", None)
    featurizer_config_file = train_config_file.split(".json")[0] + "_featurizer.json"
    with open(featurizer_config_file, "w") as fp:
        json.dump(train_config, fp)
    return featurizer_config_file


def run_hti(
    hti_config: str,
    torch_device: str,
    dataloader_num_workers: int,
    run_name: str,
    logs_dir: Optional[str] = None,
):
    device_id = 0
    if torch_device == "cpu":
        device_id = -1
    elif torch_device == "gpu":
        gpu_ids = [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device_id = gpu_ids[0]

    os.makedirs(logs_dir, exist_ok=True)

    proc = subprocess.run(
        [
            "hti-cnn",
            "--config",
            hti_config,
            "--gpu",
            str(device_id),
            "--j",
            str(dataloader_num_workers),
            "--run_name",
            run_name,
        ],
        stdout=open(os.path.join(logs_dir, "hti-cnn.log"), "w")
        if logs_dir
        else subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        raise HTIError(f"HTI CNN failed with: \n {proc.stderr.decode()}")


def run_hti_featurizer(
    hti_config: str,
    torch_device: str,
    dataloader_num_workers: int,
    checkpoint_dir: str,
    logs_dir: Optional[str] = None,
):
    device_id = 0
    if torch_device == "cpu":
        device_id = -1
    elif torch_device == "gpu":
        gpu_ids = [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device_id = gpu_ids[0]

    os.makedirs(logs_dir, exist_ok=True)

    proc = subprocess.run(
        [
            "hti-cnn-feature",
            "--config",
            hti_config,
            "--gpu",
            str(device_id),
            "--j",
            str(dataloader_num_workers),
            "--checkpoint",
            os.path.join(checkpoint_dir),
        ],
        stdout=open(os.path.join(logs_dir, "hti-cnn_featurizer.log"), "w")
        if logs_dir
        else subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        raise HTIError(f"HTI CNN failed with: \n {proc.stderr.decode()}")


def preprocess_features(results_dir: str, hti_index_file: str):
    features = np.load(
        os.path.join(results_dir, "ensemble-features-val.npz"), allow_pickle=True
    )
    hti_index_file = pd.read_csv(hti_index_file).rename(
        {"SAMPLE_KEY": "granular_index"}, axis=1
    )
    hti_index_file["index"] = hti_index_file.apply(
        lambda x: x["granular_index"][:-2], axis=1
    )

    hti_index_file = hti_index_file[
        hti_index_file.granular_index.isin(features["granular_ids"])
    ]
    hti_index_file[["input_compound_id", "smiles"]].drop_duplicates().to_csv(
        os.path.join(results_dir, f"{T2_IMAGES}.csv"), index=False
    )

    granular_features_df = pd.DataFrame(features["granular_features"].mean(axis=0))
    granular_features_df["granular_index"] = features["granular_ids"]
    granular_features_df = (
        hti_index_file[["input_compound_id", "granular_index"]]
        .merge(granular_features_df, on="granular_index", how="inner")
        .drop(["granular_index"], axis=1)
    )

    features_df = pd.DataFrame(features["features"])
    features_df["index"] = features["ids"]
    features_df = (
        hti_index_file[["input_compound_id", "index"]]
        .drop_duplicates()
        .merge(features_df, on="index", how="inner")
        .drop(["index"], axis=1)
    )

    granular_features_df.to_csv(
        os.path.join(results_dir, "T_image_features_granular.csv"), index=False
    )
    features_df.to_csv(
        os.path.join(results_dir, f"{T_IMAGE_FEATURES}.csv"), index=False
    )
