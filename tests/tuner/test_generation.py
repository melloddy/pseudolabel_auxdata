import os.path

from pseudolabel.tuner import generation


def test_filter_data_by_id(config):
    generation.find_overlap_with_melloddy(
        t0_melloddy_path=config.t0_melloddy_path,
        t1_melloddy_path=config.t1_melloddy_path,
        t2_images_path=config.t2_images_path,
        tuner_images_input_folder=config.tuner_input_folder_image,
    )
    generation.synchronize_thresholds(
        t8_tunner_path=config.t8c_baseline,
        tuner_images_input_folder=config.tuner_input_folder_image,
        delete_intermediate_files=False,
    )
