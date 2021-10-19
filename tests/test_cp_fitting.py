import os

from pseudolabel.cp_fitting import splitting_data, fit_cp, generate_task_stats
from pseudolabel.pseudolabel_config import PseudolabelConfig


def test_splitting_data(config: PseudolabelConfig):
    splitting_data(config.tuner_output_folder, config.intermediate_files_folder)


def test_split_and_fit_cp(config: PseudolabelConfig):
    preds_fva, labels, cdvi_fit, cdvi_eval = splitting_data(
        config.tuner_output_folder, config.intermediate_files_folder
    )

    fit_cp(
        preds_fva=preds_fva,
        labels=labels,
        cdvi_fit=cdvi_fit,
        cdvi_eval=cdvi_eval,
        analysis_folder=config.analysis_folder,
    )


def test_generate_task_stats(config: PseudolabelConfig):
    generate_task_stats(config.analysis_folder)

