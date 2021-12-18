import json
import logging
import os

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def generate_t_aux_pl(
    intermediate_files_folder: str, t2_melloddy_path: str, t2_images_path: str
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

    # T0 creation

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

    aux_data_dir = os.path.join(intermediate_files_folder, "aux_data_no_labels")
    os.makedirs(aux_data_dir, exist_ok=True)

    t0_image.to_csv(
        os.path.join(aux_data_dir, "T0_image_pseudolabels_no_labels.csv"), index=False
    )

    # T1 export
    t1["standard_value"] = np.where(t1["standard_value"] == 1, 1, -1)

    t1["input_assay_id_melloddy"] = t1["input_assay_id"].map(image_cont_iai_to_new_iai)

    t1_image_out = t1.drop(columns=["input_assay_id"]).rename(
        columns={"input_assay_id_melloddy": "input_assay_id"}
    )

    t1_image_out.to_csv(
        os.path.join(aux_data_dir, "T1_image_pseudolabels_no_labels.csv"), index=False
    )

    t2_images = pd.read_csv(t2_images_path)
    t2_melloddy = pd.read_csv(t2_melloddy_path)

    t2_images_pseudolabels = (
        pd.concat([t2_images, t2_melloddy])
        .groupby(by="input_compound_id")
        .first()
        .reset_index()
    )
    t2_images_pseudolabels.to_csv(
        os.path.join(aux_data_dir, "T2_image_pseudolabels_no_labels.csv"), index=False
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
    # retrieve the input compound ids from T5

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
        os.path.join(
            intermediate_files_folder, "aux_data_no_labels", "T1_image_truelabels.csv"
        ),
        index=False,
    )


def replace_pseudolabels_w_labels(intermediate_files_folder: str):
    path = os.path.join(
        intermediate_files_folder,
        "aux_data_no_labels",
        "T1_image_pseudolabels_no_labels.csv",
    )
    t1_image = pd.read_csv(path)
    path = os.path.join(
        intermediate_files_folder,
        "aux_data_no_labels",
        "T0_image_pseudolabels_no_labels.csv",
    )
    t0_image = pd.read_csv(path)
    path = os.path.join(
        intermediate_files_folder,
        "aux_data_no_labels",
        "T2_image_pseudolabels_no_labels.csv",
    )
    t2_image = pd.read_csv(path)

    path = os.path.join(
        intermediate_files_folder, "aux_data_no_labels", "T1_image_truelabels.csv"
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

    path = os.path.join(
        intermediate_files_folder, "mapping", "baseline_image_model_task_mapping.csv"
    )
    t_parent_id = pd.read_csv(
        path, usecols=["input_assay_id", "baseline_compliant_input_assay_id_image"]
    ).rename(
        {
            "input_assay_id": "parent_assay_id",
            "baseline_compliant_input_assay_id_image": "input_assay_id",
        },
        axis=1,
    )

    t0_image["use_in_regression"] = False
    t0_image["direction"] = "high"
    t0_image["is_binary"] = True
    t0_image["catalog_assay_id"] = np.nan
    t0_image = t0_image.merge(t_parent_id, on="input_assay_id", how="right")

    for i in range(1, 6):
        t0_image[f"expert_threshold_{i}"] = np.nan

    t0_image = t0_image[
        [
            "input_assay_id",
            "assay_type",
            "is_binary",
            "use_in_regression",
            "expert_threshold_1",
            "expert_threshold_2",
            "expert_threshold_3",
            "expert_threshold_4",
            "expert_threshold_5",
            "direction",
            "catalog_assay_id",
            "parent_assay_id",
        ]
    ]

    t0_image = t0_image.astype(
        dtype={
            "input_assay_id": "int64",
            "assay_type": "str",
            "use_in_regression": "bool",
            "is_binary": "bool",
            "expert_threshold_1": "float64",
            "expert_threshold_2": "float64",
            "expert_threshold_3": "float64",
            "expert_threshold_4": "float64",
            "expert_threshold_5": "float64",
            "direction": "str",
            "catalog_assay_id": "float64",
            "parent_assay_id": "float64",
        }
    )

    t1_images_preds_w_true = t1_images_preds_w_true.astype(
        dtype={
            "input_compound_id": "int64",
            "input_assay_id": "int64",
            "standard_qualifier": "str",
            "standard_value": "float64",
        }
    )

    t2_image = t2_image.astype(dtype={"input_compound_id": "int64", "smiles": "str"})

    t0_images_pseudolabels = t0_image[
        t0_image.input_assay_id.isin(t1_images_preds_w_true.input_assay_id.unique())
    ]

    t1_images_pseudolabels = t1_images_preds_w_true[
        t1_images_preds_w_true.input_compound_id.isin(
            t2_image.input_compound_id.unique()
        )
    ]
    t2_images_pseudolabels = t2_image[
        t2_image.input_compound_id.isin(
            t1_images_pseudolabels.input_compound_id.unique()
        )
    ]

    os.makedirs(os.path.join(intermediate_files_folder, "aux_data_full"), exist_ok=True)

    t1_images_pseudolabels.to_csv(
        os.path.join(
            intermediate_files_folder,
            "aux_data_full",
            "T1_images_pseudolabels_full.csv",
        ),
        index=False,
    )

    t0_images_pseudolabels.to_csv(
        os.path.join(
            intermediate_files_folder,
            "aux_data_full",
            "T0_images_pseudolabels_full.csv",
        ),
        index=False,
    )

    t2_images_pseudolabels.to_csv(
        os.path.join(
            intermediate_files_folder,
            "aux_data_full",
            "T2_images_pseudolabels_full.csv",
        ),
        index=False,
    )


def filter_low_confidence_pseudolabels(
    intermediate_files_folder: str,
    output_pseudolabel_folder: str,
    analysis_folder: str,
    threshold: float,
):
    aux_data_full_path = os.path.join(intermediate_files_folder, "aux_data_full")
    cp_path = os.path.join(analysis_folder, "cp")
    mapping_path = os.path.join(intermediate_files_folder, "mapping")

    df_cp = pd.read_csv(os.path.join(cp_path, "summary_eps_0.05.csv"))
    baseline_imagemodel_mapping = pd.read_csv(
        os.path.join(mapping_path, "baseline_image_model_task_mapping.csv")
    )
    df_cp_mgd = pd.merge(
        df_cp,
        baseline_imagemodel_mapping,
        left_on="index",
        right_on="cont_classification_task_id_image",
        how="inner",
    )

    df_cp_mgd.to_csv(os.path.join(cp_path, "summary_eps_0.05_mgd_cp.csv"), index=False)

    T0_aux = pd.read_csv(
        os.path.join(aux_data_full_path, "T0_images_pseudolabels_full.csv")
    )
    df_matching = pd.merge(
        df_cp_mgd,
        T0_aux,
        left_on="baseline_compliant_input_assay_id_image",
        right_on="input_assay_id",
        how="inner",
        suffixes=("_image", "_aux"),
    )

    iai_to_keep = set(
        df_matching.query("NPV_0 > @threshold")
        .query("PPV_1 > @threshold")["input_assay_id_aux"]
        .dropna()
    )
    T0_aux_filtered = T0_aux[T0_aux.input_assay_id.isin(iai_to_keep)]

    T1_aux = pd.read_csv(
        os.path.join(aux_data_full_path, "T1_images_pseudolabels_full.csv")
    )
    T2_aux = pd.read_csv(
        os.path.join(aux_data_full_path, "T2_images_pseudolabels_full.csv")
    )

    T1_aux_filtered = T1_aux[T1_aux.input_assay_id.isin(T0_aux_filtered.input_assay_id)]
    T2_aux_filtered = T2_aux[
        T2_aux.input_compound_id.isin(T1_aux_filtered.input_compound_id)
    ]

    os.makedirs(output_pseudolabel_folder, exist_ok=True)
    T0_aux_filtered.to_csv(
        os.path.join(output_pseudolabel_folder, "T0_images_pseudolabels.csv"),
        index=False,
    )
    T1_aux_filtered.to_csv(
        os.path.join(output_pseudolabel_folder, "T1_images_pseudolabels.csv"),
        index=False,
    )
    T2_aux_filtered.to_csv(
        os.path.join(output_pseudolabel_folder, "T2_images_pseudolabels.csv"),
        index=False,
    )
