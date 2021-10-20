import os

from pytest import fixture

from pseudolabel.pseudolabel_config import PseudolabelConfig


@fixture(scope="session")
def data_folder() -> str:
    return "/Users/tmp2/Desktop/iktos/pseudolabel_auxdata/data/"


@fixture(scope="session")
def output_folder(data_folder) -> str:
    return os.path.join(data_folder, "output")


@fixture(scope="session")
def params_folder(data_folder) -> str:
    return os.path.join(data_folder, "params")


@fixture(scope="session")
def config(data_folder, params_folder, output_folder) -> PseudolabelConfig:
    melloddy_path = os.path.join(data_folder, "Melloddy")

    return PseudolabelConfig(
        t0_melloddy_path=os.path.join(melloddy_path, "input", "T0.csv"),
        t1_melloddy_path=os.path.join(melloddy_path, "input", "T1.csv"),
        t2_melloddy_path=os.path.join(melloddy_path, "input", "T2.csv"),
        t2_images_path=os.path.join(data_folder, "T2_images.csv"),
        t_images_features_path=os.path.join(data_folder, "T_image_features_std.csv"),
        key_json=os.path.join(params_folder, "key.json"),
        parameters_json=os.path.join(params_folder, "parameters.json"),
        ref_hash_json=os.path.join(params_folder, "ref_hash.json"),
        tuner_output_folder_baseline=os.path.join(melloddy_path, "tuner_output"),
        output_folder_path=output_folder,
        sparsechem_path="/Users/tmp2/Desktop/iktos/sparsechem",
        dataloader_num_workers=0,
    )


@fixture(scope="session")
def tuner_images_output_folder():
    return "/Users/tmp2/Desktop/iktos/pseudolabel_auxdata/data/output"
