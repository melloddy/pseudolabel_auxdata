from pseudolabel.tuner import tuner_tools
import os


def test_run_tuner(config):
    tuner_tools.run_tuner(
        tuner_images_input_folder=config.tuner_input_folder_image,
        json_params=config.parameters_json,
        json_key=config.key_json,
        json_ref_hash=config.ref_hash_json,
        output_dir=config.output_folder_path,
        n_cpu=config.max_cpu,
    )


def test_process_tuner_output(config):
    tuner_tools.process_tuner_output(
        tuner_output_images=config.tuner_output_folder_image,
        image_features_file=config.t_images_features_path,
    )
