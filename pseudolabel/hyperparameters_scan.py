import logging
import os
import subprocess
from typing import List, Tuple

from tqdm import tqdm

from pseudolabel.constants import IMAGE_MODEL_NAME
from pseudolabel.errors import HyperOptError

LOGGER = logging.getLogger(__name__)


def run_hyperopt(
    epoch_lr_steps: List[Tuple[int, int]],
    hidden_sizes: List[str],
    dropouts: List[float],
    hp_output_dir: str,
    sparsechem_trainer_path: str,
    tuner_output_dir: str,
    torch_device: str,
    show_progress: bool = True,
):
    # TODO add step number/steps

    distqdm = not show_progress
    # Loop over hyperparameter combinations and edit script
    i = 0
    for epoch_lr_step in tqdm(
        epoch_lr_steps,
        desc="HyperOpt Epoch - LR step",
        disable=distqdm,
    ):
        for dropout in tqdm(
            dropouts, desc="HyperOpt dropout", disable=distqdm, leave=False
        ):
            for hidden in tqdm(
                hidden_sizes,
                desc="HyperOpt Hidden Size",
                disable=distqdm,
                leave=False,
            ):
                i += 1
                num = str(i).zfill(3)

                # Remove spaces in hidden layers (for file name)
                hidden_name = hidden.replace(" ", "-")
                run_name = f"Run_{num}_epoch_lr_step_{epoch_lr_step[0]}_{epoch_lr_step[1]}_drop_{dropout}_size_{hidden_name}"
                # Create script folder and create script

                current_model_dir = os.path.join(hp_output_dir, run_name)
                os.makedirs(current_model_dir, exist_ok=True)

                log_file = os.path.join(current_model_dir, "log.txt")
                # TODO Check best way to call subprocess
                # TODO Add all arguments of sparsechem

                LOGGER.info(
                    f"Running HyperOpt with Hidden={hidden} Dropout={dropout} EpochLRStep={epoch_lr_step}"
                )
                proc = subprocess.run(
                    [
                        "python",
                        sparsechem_trainer_path,
                        "--x",
                        os.path.join(
                            tuner_output_dir,
                            "matrices",
                            "cls",
                            "cls_T11_x_features.npz",
                        ),
                        "--y",
                        os.path.join(
                            tuner_output_dir, "matrices", "cls", "cls_T10_y.npz"
                        ),
                        "--folding",
                        os.path.join(
                            tuner_output_dir,
                            "matrices",
                            "cls",
                            "cls_T11_fold_vector.npy",
                        ),
                        "--weights_class",
                        os.path.join(
                            tuner_output_dir,
                            "matrices",
                            "cls",
                            "cls_weights.csv",
                        ),
                        "--hidden_sizes",
                        hidden,
                        "--last_dropout",
                        str(dropout),
                        "--middle_dropout",
                        str(dropout),
                        "--last_non_linearity",
                        "relu",
                        "--non_linearity",
                        "relu",
                        "--input_transform",
                        "none",
                        "--lr",
                        "0.001",
                        "--lr_alpha",
                        "0.3",
                        "--lr_steps",
                        str(epoch_lr_step[1]),
                        "--epochs",
                        str(epoch_lr_step[0]),
                        "--normalize_loss",
                        "100_000",
                        "--eval_frequency",
                        "1",
                        "--batch_ratio",
                        "0.02",
                        "--fold_va",
                        "2",
                        "--fold_te",
                        "0",
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
