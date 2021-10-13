from pseudolabel.tuner import tuner_tools
import os


def test_run_tuner(output_folder, params_folder):
    tuner_tools.run_tuner(
        images_input_folder=output_folder,
        json_params=os.path.join(params_folder, "parameters.json"),
        json_key=os.path.join(params_folder, "key.json"),
        json_ref_hash=os.path.join(params_folder, "ref_hash.json"),
        output_dir=output_folder,
    )


# if __name__ == "__main__":
#     test_run_tuner(
#         "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/output",
#         "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/params",
#         "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/output/tuner_images",
#     )


def test_process_tuner_output(tuner_images_output_folder, config):
    tuner_tools.process_tuner_output(
        tuner_images_output_folder, image_features_file=config.t_images_features_path
    )
