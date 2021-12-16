import logging
import os
from typing import Optional

from pseudolabel.pipeline_steps import (
    apply_cp_step,
    data_preprocessing_step,
    fitting_cp_step,
    generate_all_pred_step,
    generate_T_files_pl_step,
    image_model_hyperscan_step,
)
from pseudolabel.pseudolabel_config import PseudolabelConfig

LOGGER = logging.getLogger(__name__)


class PseudolabelPipe:
    def __init__(self, config_file: str = None):
        self.config: Optional[PseudolabelConfig] = None
        self.steps_dict = {
            "data_preprocessing": data_preprocessing_step,
            "image_model_hyperscan": image_model_hyperscan_step,
            "fit_conformal_predictors": fitting_cp_step,
            "generate_all_predictions": generate_all_pred_step,
            "apply_cp_aux": apply_cp_step,
            "generate_T_files_pseudolabels": generate_T_files_pl_step,
        }

        self.steps_dict = self.add_index_steps()
        if config_file:
            self.load_config(config_file)

    def add_index_steps(self):
        indexed_steps = {}
        for i, step in enumerate(self.steps_dict):
            indexed_steps[step] = (i, self.steps_dict[step])
        return indexed_steps

    def load_config(self, config_file: str):

        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file is not found : {config_file}")

        self.config = PseudolabelConfig.load_config(config_file)

    def run_partial_pipe(self, starting_step: str):
        if starting_step not in self.steps_dict:
            raise ValueError(
                f"{starting_step} is not a valid starting step, please choose one on the following : "
                f"{list(self.steps_dict.keys())}"
            )

        starting_step_ind = self.steps_dict[starting_step][0]

        for step in self.steps_dict:
            if self.steps_dict[step][0] < starting_step_ind:
                continue
            self.steps_dict[step][1](self.config)

    def run_full_pipe(self):
        for step in self.steps_dict:
            self.steps_dict[step][1](self.config)
        LOGGER.info("Pipeline finished")
