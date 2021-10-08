import os.path

from pseudolabel.tuner import generation


def test_filter_data_by_id(images_folder):
    input_folder = "../../data/Melloddy"
    generation.find_overlap_with_melloddy(
        os.path.join(input_folder, "T0.csv"),
        os.path.join(input_folder, "T1.csv"),
        os.path.join(input_folder, "T2.csv"),
        images_folder,
    )

    generation.synchronize_thresholds(
        os.path.join("../../data/T8c.csv"), tuner_input_images=images_folder
    )
