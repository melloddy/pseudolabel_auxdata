from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd
import torch

from pseudolabel import utils
from pseudolabel.constants import (
    DEFAULT_DROPOUTS,
    DEFAULT_EPOCHS_LR_STEPS,
    DEFAULT_HIDDEN_SIZES,
    IMAGE_MODEL_DATA,
)


@dataclass
class PseudolabelConfig:
    t0_melloddy_path: str
    t1_melloddy_path: str
    t2_melloddy_path: str
    t2_images_path: str
    t_images_features_path: str

    key_json: str
    parameters_json: str
    ref_hash_json: str

    sparsechem_path: str

    tuner_output_folder_baseline: str
    output_folder_path: str

    max_cpu: int = multiprocessing.cpu_count() - 1
    dataloader_num_workers: int = multiprocessing.cpu_count() - 1
    torch_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pseudolabel_threshold: float = 0.9
    min_threshold: float = 0.9

    imagemodel_hidden_size: List[List[str]] = field(
        default_factory=DEFAULT_HIDDEN_SIZES
    )
    imagemodel_epochs_lr_steps: List[Tuple[int, int]] = field(
        default_factory=DEFAULT_EPOCHS_LR_STEPS
    )
    imagemodel_dropouts: List[float] = field(default_factory=DEFAULT_DROPOUTS)

    show_progress: bool = True

    @property
    def t8c_baseline(self) -> str:
        path = os.path.join(
            self.tuner_output_folder_baseline,
            "results_tmp",
            "classification",
            "T8c.csv",
        )
        return path

    @property
    def t10c_cont_baseline(self) -> str:
        path = os.path.join(
            self.tuner_output_folder_baseline,
            "results",
            "T10c_cont.csv",
        )
        return path

    @property
    def t5_baseline(self) -> str:
        path = os.path.join(
            self.tuner_output_folder_baseline,
            "mapping_table",
            "T5.csv",
        )
        return path

    @property
    def tuner_output_folder_image(self) -> str:
        path = os.path.join(self.output_folder_path, IMAGE_MODEL_DATA, "tuner_output")
        return path

    @property
    def tuner_input_folder_image(self) -> str:
        path = os.path.join(self.output_folder_path, IMAGE_MODEL_DATA, "tuner_input")
        return path

    @property
    def hyperopt_output_folder(self) -> str:
        path = os.path.join(self.output_folder_path, "hyperopt")
        return path

    @property
    def intermediate_files_folder(self) -> str:
        path = os.path.join(self.output_folder_path, "intermediate")
        return path

    @property
    def log_dir(self) -> str:
        return os.path.join(self.output_folder_path, "logs")

    @property
    def analysis_folder(self) -> str:
        path = os.path.join(self.output_folder_path, "analysis")
        return path

    @property
    def output_pseudolabel_folder(self) -> str:
        path = os.path.join(self.output_folder_path, "pseudolabels")
        return path

    @property
    def sparsechem_trainer_path(self) -> str:
        return os.path.join(self.sparsechem_path, "examples", "chembl", "train.py")

    @property
    def sparsechem_predictor_path(self) -> str:
        return os.path.join(self.sparsechem_path, "examples", "chembl", "predict.py")

    def __check_files_exist(self):
        utils.check_file_exists(self.t0_melloddy_path)
        utils.check_file_exists(self.t1_melloddy_path)
        utils.check_file_exists(self.t2_melloddy_path)

        utils.check_file_exists(self.t2_images_path)
        utils.check_file_exists(self.t_images_features_path)

        utils.check_file_exists(self.t8c_baseline)
        utils.check_file_exists(self.t5_baseline)
        utils.check_file_exists(self.t10c_cont_baseline)

        utils.check_file_exists(self.key_json)
        utils.check_file_exists(self.parameters_json)
        utils.check_file_exists(self.ref_hash_json)

    def __t_images_check(self):
        t_images = pd.read_csv(self.t_images_features_path)
        assert (
            sum(t_images.input_compound_id.duplicated()) == 0
        ), f"{self.t_images_features_path} can't have duplicates"

    def __check_sparsechem(self):
        # TODO
        pass

    def __check_threshold(self):
        assert self.pseudolabel_threshold > self.min_threshold, (
            f"pseudolabel quality threshold {self.pseudolabel_threshold} should be greater than the minimum threshold"
            f" value {self.min_threshold}"
        )

    def check_data(self):
        # TODO Add more checks
        self.__check_files_exist()
        self.__t_images_check()
        self.__check_threshold()

    @classmethod
    def load_config(cls, json_path: str) -> PseudolabelConfig:
        with open(json_path) as config_file:
            config_dict = json.load(config_file)
            config = cls(**config_dict)
            config.check_data()
            return config
