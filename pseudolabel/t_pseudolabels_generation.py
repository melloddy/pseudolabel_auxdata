import json
import logging
import os
import shutil

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def generate_t_aux_pl(
    intermediate_files_folder: str, t0_melloddy_path: str, t2_images_path: str
):
    # add labels
    # integerization of compound id
    # integerization and mapping of the assay id
    # T2, T0 creation
    # ---

    t1 = pd.read_csv(
        os.path.join(
            intermediate_files_folder,
            "image_pseudolabel_aux_nolabels/T1_image_pseudolabel_aux_nolabels.csv",
        )
    )

    ## T0 creation

    # change from AUX_IMG to AUX_HTS
    # change the 1.0 label to 6.0 (to accommodate the rscore 3 threshold and force positive labels)
    # remove the 0.5 threshold from T0

    n_image_id_start = 10000000  # (high number)\n",
    image_cont_iai_to_new_iai = {
        e: i + n_image_id_start for i, e in enumerate(t1.input_assay_id.unique())
    }

    mapping_folder_path = os.path.join(intermediate_files_folder, "mapping")
    os.makedirs(mapping_folder_path, exist_ok=True)

    with open(
        os.path.join(mapping_folder_path, "image_cont_iai_to_new_iai.json"), "w"
    ) as fp:
        json.dump({str(k): v for k, v in image_cont_iai_to_new_iai.items()}, fp)

    t0_image = pd.DataFrame.from_dict(
        image_cont_iai_to_new_iai, orient="index"
    ).reset_index()
    t0_image = t0_image.rename(columns={"index": "image_col_idx", 0: "input_assay_id"})

    t0_image["source"] = "image-based pseudolabels"
    t0_image["assay_type"] = "AUX_PL"
    t0_image["assay_type_reason"] = "image-based pseudolabels"
    t0_image["model_description"] = "img_" + t0_image["image_col_idx"].astype(str)
    t0_image["assay_description"] = "img_" + t0_image["image_col_idx"].astype(str)

    aux_data_dir = os.path.join(intermediate_files_folder, "aux_data")
    os.makedirs(aux_data_dir, exist_ok=True)

    t0_image.to_csv(
        os.path.join(aux_data_dir, "T0_image_pseudolabels_full.csv"), index=False
    )

    # T1 export
    t1["standard_value"] = np.where(t1["standard_value"] == 1, 1, -1)

    t1["input_assay_id_melloddy"] = t1["input_assay_id"].map(image_cont_iai_to_new_iai)

    t1_image_out = t1.drop(columns=["input_assay_id"]).rename(
        columns={"input_assay_id_melloddy": "input_assay_id"}
    )

    t1_image_out.to_csv(
        os.path.join(aux_data_dir, "T1_image_pseudolabels_full.csv"), index=False
    )

    shutil.copyfile(
        os.path.join(t2_images_path),
        os.path.join(aux_data_dir, "T2_image_pseudolabels_full.csv"),
    )


def find_labels_of_auxtasks(
    tuner_output_folder_baseline: str,
    tuner_output_folder_image: str,
    intermediate_files_folder: str,
):
    # take the labels from the y matrix, then put them into T1 csv format in the aux tasks corresponding to the main
    # tasks. In the next step/notebook we will join them together with the other data (melloddy/pseudolabels)
    # The challenge is to find the corresponding auxiliary task
    # cont_classification_task_id (baseline) -> iai + threshold -> cont_classification_task_id (image_model) -> melloddy_compatible iai

    t8c_baseline_path = os.path.join(
        tuner_output_folder_baseline, "results_tmp", "classification", "T8c.csv"
    )

    t8_baseline = pd.read_csv(t8c_baseline_path)
    t8_baseline["threshold"] = t8_baseline["threshold"].round(decimals=5)

    t8c_image_path = os.path.join(
        tuner_output_folder_image, "results_tmp", "classification", "T8c.csv"
    )
    t8_imagemodel = pd.read_csv(t8c_image_path)
    t8_imagemodel["threshold"] = t8_imagemodel["threshold"].round(decimals=5)

    t8_mgd = pd.merge(
        t8_imagemodel,
        t8_baseline,
        how="inner",
        on=["input_assay_id", "threshold"],
        suffixes=("_image", "_baseline"),
    )

    t8_mgd = t8_mgd[~t8_mgd["cont_classification_task_id_image"].isna()]

    # some auxiliary thresholds will still be added in addition to the expert thresholds
    # therefore, not all of the image model tasks correspond to a baseline task

    mapping_folder = os.path.join(intermediate_files_folder, "mapping")
    with open(os.path.join(mapping_folder, "image_cont_iai_to_new_iai.json")) as fp:
        image_cont_iai_to_new_iai = json.load(fp)

    t8_mgd["baseline_compliant_input_assay_id_image"] = (
        t8_mgd["cont_classification_task_id_image"]
        .astype(int)
        .astype(str)
        .map(image_cont_iai_to_new_iai)
    )

    t8_mgd = t8_mgd[~t8_mgd["baseline_compliant_input_assay_id_image"].isna()]

    t8_mgd.to_csv(
        os.path.join(mapping_folder, "baseline_image_model_task_mapping.csv"),
        index=False,
    )
    # now filter the baseline T10c on the iais which can actually be mapped from t8_mgd
    ## retrieve the input compound ids from T5

    path = os.path.join(tuner_output_folder_baseline, "results", "T10c_cont.csv")
    t10c_baseline = pd.read_csv(path)
    path = os.path.join(tuner_output_folder_baseline, "mapping_table", "T5.csv")
    t5_baseline = pd.read_csv(path)

    # filter on relevant tasks

    t10c_baseline_lim = pd.merge(
        t10c_baseline,
        pd.merge(t8_baseline, t8_mgd, on=["input_assay_id", "threshold"], how="inner"),
        on="cont_classification_task_id",
        how="inner",
    )

    t10c_baseline_lim2 = pd.merge(
        t5_baseline[["input_compound_id", "descriptor_vector_id"]].drop_duplicates(),
        t10c_baseline_lim,
        on="descriptor_vector_id",
        how="inner",
    )

    t10c_baseline_lim2["standard_value"] = t10c_baseline_lim2["class_label"]

    t10c_baseline_lim2["standard_qualifier"] = "="

    df_t1_truelabels_out = t10c_baseline_lim2[
        [
            "baseline_compliant_input_assay_id_image",
            "input_compound_id",
            "standard_qualifier",
            "standard_value",
        ]
    ].drop_duplicates()

    df_t1_truelabels_out.to_csv(
        os.path.join(intermediate_files_folder, "aux_data", "T1_image_truelabels.csv"),
        index=False,
    )


def replace_pseudolabels_w_labels(
    intermediate_files_folder: str, output_pseudolabel_folder: str
):

    path = os.path.join(
        intermediate_files_folder, "aux_data", "T1_image_pseudolabels_full.csv"
    )
    t1_image = pd.read_csv(path)
    path = os.path.join(
        intermediate_files_folder, "aux_data", "T0_image_pseudolabels_full.csv"
    )
    t0_image = pd.read_csv(path)
    path = os.path.join(
        intermediate_files_folder, "aux_data", "T2_image_pseudolabels_full.csv"
    )
    t2_image = pd.read_csv(path)

    path = os.path.join(
        intermediate_files_folder, "aux_data", "T1_image_truelabels.csv"
    )
    t1_image_true = pd.read_csv(path)

    t1_images_preds_w_true = pd.concat(
        [
            t1_image_true.rename(
                columns={"baseline_compliant_input_assay_id_image": "input_assay_id"}
            ),
            t1_image,
        ]
    )

    t1_images_preds_w_true = (
        t1_images_preds_w_true.groupby(by=["input_compound_id", "input_assay_id"])
        .first()
        .reset_index()
    )

    t0_image["use_in_regression"] = False
    t0_image["direction"] = "high"
    for i in range(1, 6):
        t0_image[f"expert_threshold_{i}"] = np.nan

    t0_image = t0_image[
        [
            "input_assay_id",
            "assay_type",
            "use_in_regression",
            "expert_threshold_1",
            "expert_threshold_2",
            "expert_threshold_3",
            "expert_threshold_4",
            "expert_threshold_5",
            "direction",
        ]
    ]

    t0_images_pseudolabels = t0_image[
        t0_image.input_assay_id.isin(t1_images_preds_w_true.input_assay_id)
    ]
    t2_images_pseudolabels = t2_image[
        t2_image.input_compound_id.isin(t1_images_preds_w_true.input_compound_id)
    ]

    os.makedirs(output_pseudolabel_folder, exist_ok=True)

    t1_images_preds_w_true.to_csv(
        os.path.join(output_pseudolabel_folder, "T1_images_pseudolabels.csv"),
        index=False,
    )

    t0_images_pseudolabels.to_csv(
        os.path.join(output_pseudolabel_folder, "T0_images_pseudolabels.csv"),
        index=False,
    )

    t2_images_pseudolabels.to_csv(
        os.path.join(output_pseudolabel_folder, "T2_images_pseudolabels.csv"),
        index=False,
    )
