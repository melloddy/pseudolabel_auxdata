import logging
import os

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def find_overlap_with_melloddy(
    *,
    t0_melloddy_path: str,
    t1_melloddy_path: str,
    t2_images_path: str,
    output_folder: str,
):
    LOGGER.info("Filtering input data on compound_id")
    os.makedirs(output_folder, exist_ok=True)
    t0_melloddy = pd.read_csv(t0_melloddy_path)
    t1_melloddy = pd.read_csv(t1_melloddy_path)
    t2_all_images = pd.read_csv(t2_images_path)

    # prepare T1
    t1_images = t1_melloddy[
        t1_melloddy.input_compound_id.isin(t2_all_images.input_compound_id)
    ]
    t1_images.to_csv(os.path.join(output_folder, "T1.csv"), index=False)

    # filtering compounds without input_assay_id
    t2_images = t2_all_images[
        t2_all_images.input_compound_id.isin(t1_images.input_compound_id)
    ]
    t2_images.to_csv(os.path.join(output_folder, "T2.csv"), index=False)

    # prepare T0
    t0_images = t0_melloddy[t0_melloddy.input_assay_id.isin(t1_images.input_assay_id)]
    t0_images.to_csv(os.path.join(output_folder, "T0.csv"), index=False)


def synchronize_thresholds(
    t8_tunner_path: str, tuner_input_images: str, delete_intermediate_files: bool = True
):
    LOGGER.info("Synchronizing thresholds")
    # T0-file csv tuner input creation for the step1 image model
    t0_path = os.path.join(tuner_input_images, "T0.csv")
    t0 = pd.read_csv(t0_path)

    t8 = pd.read_csv(
        t8_tunner_path, usecols=["input_assay_id", "threshold", "threshold_method"]
    ).rename(columns={"threshold": "expert_threshold"})
    t8["subgroup_index"] = t8.groupby("input_assay_id").cumcount() + 1

    df = pd.pivot_table(
        t8.drop_duplicates(),
        columns=["subgroup_index"],
        values=["expert_threshold"],
        index=["input_assay_id"],
    )
    df.columns = df.columns.map(lambda x: "_".join([*map(str, x)]))

    threshold_cols = [f"expert_threshold_{i}" for i in range(1, 6)]
    t0_out = pd.merge(
        t0.drop(columns=threshold_cols), df, how="inner", on="input_assay_id"
    )

    # MT will freeze if not all the expert threshold columns are present
    for col in threshold_cols:
        if col not in t0_out.columns:
            t0_out[col] = np.nan

    t0_out.to_csv(
        os.path.join(tuner_input_images, "T0_synchedthreshold.csv"), index=False
    )

    iais_in_scope = set(t0_out["input_assay_id"])  # noqa
    t1_path = os.path.join(tuner_input_images, "T1.csv")
    t1 = pd.read_csv(t1_path)
    t1_synchedthreshold = t1.query("input_assay_id in @iais_in_scope")
    t1_synchedthreshold.to_csv(
        os.path.join(tuner_input_images, "T1_synchedthreshold.csv"), index=False
    )

    icis = set(t1_synchedthreshold["input_compound_id"])  # noqa
    t2_path = os.path.join(tuner_input_images, "T2.csv")
    t2 = pd.read_csv(t2_path)
    t2.query("input_compound_id in @icis").to_csv(
        os.path.join(tuner_input_images, "T2_synchedthreshold.csv"), index=False
    )

    if delete_intermediate_files:
        LOGGER.debug(f"Remove intermediate files from {tuner_input_images}")
        os.remove(t0_path)
        os.remove(t1_path)
        os.remove(t2_path)
