from pseudolabel.tuner import tuner_output_processing


def test_process_tuner_output(tuner_images_output_folder, config):
    tuner_output_processing.process_tuner_output(
        tuner_images_output_folder, image_features_file=config.t_images_features_path
    )
