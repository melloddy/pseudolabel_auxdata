import os.path

from pseudolabel.tuner import generation


def test_filter_data_by_id(output_folder):
    input_folder = "../../data/Melloddy"
    generation.find_overlap_with_melloddy(
        t0_melloddy_path=os.path.join(input_folder, "T0.csv"),
        t1_melloddy_path=os.path.join(input_folder, "T1.csv"),
        t2_images_path=os.path.join(input_folder, "..", "T2_images.csv"),
        output_folder=output_folder,
    )

    generation.synchronize_thresholds(
        os.path.join("../../data/T8c.csv"), tuner_input_images=output_folder
    )
