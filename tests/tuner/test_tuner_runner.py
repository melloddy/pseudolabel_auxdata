import os

from pseudolabel.tuner import tuner_runner


def test_run_tuner(images_folder, params_folder, tuner_images_output_folder):
    tuner_runner.run_tuner(
        images_input_folder=images_folder,
        json_params=os.path.join(params_folder, "parameters.json"),
        json_key=os.path.join(params_folder, "key.json"),
        json_ref_hash=os.path.join(params_folder, "ref_hash.json"),
        output_dir=tuner_images_output_folder,
    )


if __name__ == "__main__":
    test_run_tuner(
        "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/output",
        "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/params",
        "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/output/tuner_images",
    )
