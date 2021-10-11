import os

from pytest import fixture

from pseudolabel.pseudolabel_config import PseudolabelConfig


@fixture(scope="session")
def data_folder() -> str:
    return "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/"


@fixture(scope="session")
def images_folder(data_folder) -> str:
    return os.path.join(data_folder, "output")


@fixture(scope="session")
def params_folder(data_folder) -> str:
    return os.path.join(data_folder, "params")


@fixture(scope="session")
def config(data_folder, params_folder) -> PseudolabelConfig:
    melloddy_path = os.path.join(data_folder, "Melloddy")

    return PseudolabelConfig(
        t0_melloddy_path=os.path.join(melloddy_path, "T0.csv"),
        t1_melloddy_path=os.path.join(melloddy_path, "T1.csv"),
        t2_melloddy_path=os.path.join(melloddy_path, "T2.csv"),
        t2_images_path=os.path.join(data_folder, "T2_images.csv"),
        t_images_features_path=os.path.join(data_folder, "T_image_features_std.csv"),
        t8c=os.path.join(data_folder, "T8c.csv"),
        key_json=os.path.join(params_folder, "key.json"),
        parameters_json=os.path.join(params_folder, "parameters.json"),
        ref_hash_json=os.path.join(params_folder, "ref_hash.json"),
    )


@fixture(scope="session")
def tuner_images_output_folder():
    return (
        "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/output/tuner_images"
    )
