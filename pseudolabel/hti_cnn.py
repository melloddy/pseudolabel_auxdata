import json
import os
import subprocess
from typing import Optional

import torch

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
