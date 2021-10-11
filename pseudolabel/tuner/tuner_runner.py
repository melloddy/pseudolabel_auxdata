import copy
import multiprocessing
import os

from melloddy_tuner.tunercli import do_prepare_training, prepare

from pseudolabel import utils
from pseudolabel.constants import TUNER_MODEL_NAME


def run_tuner(
    images_input_folder: str,
    json_params: str,
    json_key: str,
    json_ref_hash: str,
    output_dir: str,
    n_cpu: int = multiprocessing.cpu_count() - 1,
):
    args = copy.deepcopy(prepare)
    # prepare
    args.non_interactive = True

    args.structure_file = os.path.join(images_input_folder, "T2_synchedthreshold.csv")
    args.activity_file = os.path.join(images_input_folder, "T1_synchedthreshold.csv")
    args.weight_table = os.path.join(images_input_folder, "T0_synchedthreshold.csv")

    args.config_file = utils.check_file_exists(json_params)
    args.key_file = utils.check_file_exists(json_key)
    args.ref_hash = utils.check_file_exists(json_ref_hash)

    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    args.run_name = TUNER_MODEL_NAME
    args.tag = "cls"
    args.folding_method = "scaffold"
    args.number_cpu = n_cpu

    do_prepare_training(args)
