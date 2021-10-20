import os

from pseudolabel.cp_fitting import (
    splitting_data,
    fit_cp,
    generate_task_stats,
    apply_cp_aux,
)
from pseudolabel.pseudolabel_config import PseudolabelConfig


def test_splitting_data(config: PseudolabelConfig):
    splitting_data(
        tuner_output_images=config.tuner_output_folder_image,
        intermediate_files_folder=config.intermediate_files_folder,
    )


def test_split_and_fit_cp(config: PseudolabelConfig):
    preds_fva, labels, cdvi_fit, cdvi_eval = splitting_data(
        tuner_output_images=config.tuner_output_folder_image,
        intermediate_files_folder=config.intermediate_files_folder,
    )

    fit_cp(
        preds_fva=preds_fva,
        labels=labels,
        cdvi_fit=cdvi_fit,
        cdvi_eval=cdvi_eval,
        analysis_folder=config.analysis_folder,
    )


def test_generate_task_stats(config: PseudolabelConfig):
    generate_task_stats(analysis_folder=config.analysis_folder)


def test_apply_cp_aux(config: PseudolabelConfig):
    apply_cp_aux(
        analysis_folder=config.analysis_folder,
        t2_images_path=config.t2_images_path,
        intermediate_files=config.intermediate_files_folder,
    )
