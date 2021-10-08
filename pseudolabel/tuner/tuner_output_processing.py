import os

import pandas as pd
import scipy.sparse


def process_tuner_output(tuner_output_images: str, image_features_file: str):
    t5 = pd.read_csv(
        os.path.join(tuner_output_images, "mapping_table/T5.csv"),
        usecols=["input_compound_id", "descriptor_vector_id"],
    )

    t6_cont = pd.read_csv(
        os.path.join(tuner_output_images, "results/T6_cont.csv"),
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
        os.path.join(tuner_output_images, "matrices/cls/cls_T11_x.npz")
    )

    assert cls_T11_x_features.shape[0] == cls_T11_x.shape[0]

    scipy.sparse.save_npz(
        file=os.path.join(tuner_output_images, "matrices/cls/cls_T11_x_features.npz"),
        matrix=cls_T11_x_features,
    )
