import copy
import logging
import multiprocessing
import os

import pandas as pd
import scipy.sparse
from melloddy_tuner.tunercli import do_prepare_training, prepare

from pseudolabel import utils
from pseudolabel.constants import IMAGE_MODEL_NAME

LOGGER = logging.getLogger(__name__)


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
    args.run_name = IMAGE_MODEL_NAME
    args.tag = "cls"
    args.folding_method = "scaffold"
    args.number_cpu = n_cpu

    do_prepare_training(args)


def process_tuner_output(tuner_output_images: str, image_features_file: str):
    LOGGER.info("Processing tuner output")
    tuner_output_images = os.path.join(tuner_output_images, IMAGE_MODEL_NAME)
    t5 = pd.read_csv(
        os.path.join(tuner_output_images, "mapping_table", "T5.csv"),
        usecols=["input_compound_id", "descriptor_vector_id"],
    )

    t6_cont = pd.read_csv(
        os.path.join(tuner_output_images, "results", "T6_cont.csv"),
        usecols=["descriptor_vector_id", "cont_descriptor_vector_id"],
    ).drop_duplicates()

    cmpd_mapping_table = pd.merge(t5, t6_cont, on="descriptor_vector_id", how="inner")

    t_image_features = pd.read_csv(image_features_file, index_col=0)
    t_image_features.index.names = ["input_compound_id"]
    t_image_features = t_image_features.reset_index()

    x_features = pd.merge(
        t_image_features, cmpd_mapping_table, on="input_compound_id", how="right"
    )
    x_features = x_features.drop(["input_compound_id", "descriptor_vector_id"], axis=1)
    x_features = (
        x_features.groupby("cont_descriptor_vector_id").agg("mean").sort_index()
    )
    cls_T11_x_features = scipy.sparse.csr_matrix(x_features.values)

    cls_T11_x = scipy.sparse.load_npz(
        os.path.join(tuner_output_images, "matrices", "cls", "cls_T11_x.npz")
    )
    assert cls_T11_x_features.shape[0] == cls_T11_x.shape[0]

    cls_t11_features = os.path.join(
        tuner_output_images, "matrices", "cls", "cls_T11_x_features.npz"
    )

    LOGGER.debug(f"Saving {cls_t11_features} may take a few seconds...")
    scipy.sparse.save_npz(
        file=cls_t11_features,
        matrix=cls_T11_x_features,
    )
