import os.path

from pseudolabel.tuner import generation


def test_filter_data_by_id(config):
    generation.find_overlap_with_melloddy(
        t0_melloddy_path=config.t0_melloddy_path,
        t1_melloddy_path=config.t1_melloddy_path,
        t2_images_path=config.t2_images_path,
        output_folder=config.output_folder_path,
    )

    generation.synchronize_thresholds(
        config.t8c_baseline, tuner_input_images=config.output_folder_path
    )
