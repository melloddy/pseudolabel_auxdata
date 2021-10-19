import json
import os
import shutil

import numpy as np
import pandas as pd


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
