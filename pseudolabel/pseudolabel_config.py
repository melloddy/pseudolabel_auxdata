import os
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from pseudolabel import utils
from pseudolabel.constants import (
    DEFAULT_DROPOUTS,
    DEFAULT_EPOCH_LR_STEPS,
    DEFAULT_HIDDEN_SIZES,
    IMAGE_MODEL_NAME,
)


@dataclass
class PseudolabelConfig:
    t0_melloddy_path: str
    t1_melloddy_path: str
    t2_melloddy_path: str
    t2_images_path: str
    t_images_features_path: str
    t8c: str

    key_json: str
    parameters_json: str
    ref_hash_json: str

    sparsechem_path: str

    output_folder_path: str

    imagemodel_hidden_size: List[str] = field(default_factory=DEFAULT_HIDDEN_SIZES)
    imagemodel_epoch_lr_steps: List[Tuple[int, int]] = field(
        default_factory=DEFAULT_EPOCH_LR_STEPS
    )
    imagemodel_dropouts: List[float] = field(default_factory=DEFAULT_DROPOUTS)

    show_progress: bool = True

    @property
    def tuner_output_folder(self) -> str:
        path = os.path.join(self.output_folder_path, IMAGE_MODEL_NAME)
        return path

    @property
    def hyperopt_output_folder(self) -> str:
        path = os.path.join(self.output_folder_path, "hyperopt")
        return path

    def __check_files_exist(self):
        utils.check_file_exists(self.t0_melloddy_path)
        utils.check_file_exists(self.t1_melloddy_path)
        utils.check_file_exists(self.t2_melloddy_path)

        utils.check_file_exists(self.t2_images_path)
        utils.check_file_exists(self.t_images_features_path)

        utils.check_file_exists(self.t8c)

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

    def check_data(self):
        # TODO Add more checks
        self.__check_files_exist()
        self.__t_images_check()
