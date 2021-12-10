import logging
import os
import subprocess
from typing import List, Tuple
import glob
from tqdm import tqdm

from pseudolabel.constants import IMAGE_MODEL_NAME
from pseudolabel.errors import HyperOptError
from pseudolabel.errors import HyperOptCompletionError

LOGGER = logging.getLogger(__name__)

def hyperopt_completion_status(
    epochs_lr_steps: List[Tuple[int, int]],
    hidden_sizes: List[List[str]],
    dropouts: List[float],
    hp_output_dir: str,
):

    # assumes all x all
    models = glob.glob(os.path.join(hp_output_dir, "*", "*.json"))
    num_expected = len(epochs_lr_steps) * len(hidden_sizes) * len(dropouts)

    # a more gracefull alternative would be to stip the run here instead of raising an error
    # as this can tpyically be triggered by batched hyperopt runs
    if len(models) < num_expected: 
        raise HyperOptCompletionError(f"Can't find all expected models in {hp_output_dir}") 


def run_hyperopt(
    epochs_lr_steps: List[Tuple[int, int]],
    hidden_sizes: List[List[str]],
    dropouts: List[float],
    hp_output_dir: str,
    sparsechem_trainer_path: str,
    tuner_output_dir: str,
    torch_device: str,
    validation_fold: int,
    test_fold: int,
    resume_hyperopt: bool = True,
    show_progress: bool = True,
    hyperopt_subset_ind: Tuple = None,
):

    distqdm = not show_progress
    # Loop over hyperparameter combinations and edit script
    i = 0
    for epoch, lr_step in tqdm(
        epochs_lr_steps,
        desc="HyperOpt Epoch-LR Step",
        disable=distqdm,
        total=len(epochs_lr_steps),
    ):
        for dropout in tqdm(
            dropouts,
            desc="HyperOpt dropout",
            disable=distqdm,
            leave=False,
            total=len(dropouts),
        ):
            for hidden in tqdm(
                hidden_sizes,
                desc="HyperOpt Hidden Size",
                disable=distqdm,
                leave=False,
                total=len(hidden_sizes),
            ):
                i += 1
                num = str(i).zfill(3)
                if hyperopt_subset_ind and not (
                    hyperopt_subset_ind[0] <= i <= hyperopt_subset_ind[1]
                ):
                    continue

                # Remove spaces in hidden layers (for file name)
                hidden_name = "-".join(hidden)
                run_name = f"Run_{num}_epoch_{epoch}_lr_step_{lr_step}_drop_{dropout}_size_{hidden_name}"
                # Create script folder and create script

                current_model_dir = os.path.join(hp_output_dir, run_name)
                current_model_file = os.path.join(current_model_dir, IMAGE_MODEL_NAME + '.json')

                if resume_hyperopt and os.path.isfile(current_model_file):
                    continue
                os.makedirs(current_model_dir, exist_ok=True)

                log_file = os.path.join(current_model_dir, "log.txt")
                # TODO Check best way to call subprocess
                # TODO Add all arguments of sparsechem

                LOGGER.info(
                    f"Running HyperOpt with Hidden={hidden} Dropout={dropout} Epoch={epoch} LRStep={lr_step}"
                )
                proc = subprocess.run(
                    [
                        "python",
                        sparsechem_trainer_path,
                        "--x",
                        os.path.join(
                            tuner_output_dir,
                            "matrices",
                            "wo_aux",
                            "cls",
                            "cls_T11_x_features.npz",
                        ),
                        "--y_class",
                        os.path.join(
                            tuner_output_dir,
                            "matrices",
                            "wo_aux",
                            "cls",
                            "cls_T10_y.npz",
                        ),
                        "--folding",
                        os.path.join(
                            tuner_output_dir,
                            "matrices",
                            "wo_aux",
                            "cls",
                            "cls_T11_fold_vector.npy",
                        ),
                        "--weights_class",
                        os.path.join(
                            tuner_output_dir,
                            "matrices",
                            "wo_aux",
                            "cls",
                            "cls_weights.csv",
                        ),
                        "--hidden_sizes",
                        *hidden,
                        "--dropouts_trunk",
                        *[str(dropout) for i in hidden],
                        "--non_linearity",
                        "relu",
                        "--input_transform",
                        "none",
                        "--lr",
                        "0.001",
                        "--lr_alpha",
                        "0.3",
                        "--lr_steps",
                        str(lr_step),
                        "--epochs",
                        str(epoch),
                        "--normalize_loss",
                        "100_000",
                        "--eval_frequency",
                        "1",
                        "--batch_ratio",
                        "0.02",
                        "--fold_va",
                        str(validation_fold),
                        "--fold_te",
                        str(test_fold),
                        "--verbose",
                        "1",
                        "--save_model",
                        "1",
                        "--run_name",
                        IMAGE_MODEL_NAME,
                        "--output_dir",
                        current_model_dir,
                        "--dev",
                        torch_device,
                    ],
                    stdout=open(log_file, "w"),
                    stderr=subprocess.PIPE,
                )

                if proc.returncode != 0:
                    raise HyperOptError(f"HyperOpt failed: \n {proc.stderr.decode()}")
