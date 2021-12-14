import os

from pytest import fixture

from pseudolabel.pseudolabel_config import PseudolabelConfig


@fixture(scope="session")
def workdir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@fixture(scope="session")
def data_folder(workdir) -> str:
    return os.path.join(workdir, "data")


@fixture(scope="session")
def output_folder(tmpdir_factory) -> str:
    output_dir = tmpdir_factory.mktemp("output")
    return str(output_dir)


@fixture(scope="session")
def params_folder(data_folder) -> str:
    return os.path.join(data_folder, "tuner_params")


@fixture(scope="session")
def melloddy_folder(data_folder) -> str:
    return os.path.join(data_folder, "Melloddy")


@fixture(scope="session")
def hti_data_folder(data_folder) -> str:
    return os.path.join(data_folder, "pseudolabel_hti_cnn")


@fixture(scope="session")
def config(
    data_folder, melloddy_folder, params_folder, output_folder, hti_data_folder
) -> PseudolabelConfig:

    return PseudolabelConfig(
        hti_config_file=os.path.join(hti_data_folder, "gapnet-deep-ensemble-test.json"),
        hti_run_name="hti_test",
        t0_melloddy_path=os.path.join(melloddy_folder, "tuner_input", "T0.csv"),
        t1_melloddy_path=os.path.join(melloddy_folder, "tuner_input", "T1.csv"),
        t2_melloddy_path=os.path.join(melloddy_folder, "tuner_input", "T2.csv"),
        t_catalogue_path=os.path.join(melloddy_folder, "tuner_input", "T_cat.csv"),
        t2_images_path=os.path.join(data_folder, "T2_images.csv"),
        t_images_features_path=os.path.join(data_folder, "T_image_features_std.csv"),
        key_json=os.path.join(params_folder, "key.json"),
        parameters_json=os.path.join(params_folder, "parameters.json"),
        ref_hash_json=os.path.join(params_folder, "ref_hash.json"),
        tuner_output_folder_baseline=os.path.join(melloddy_folder, "tuner_output"),
        output_folder_path=output_folder,
        sparsechem_path=os.getenv(
            "SPARSECHEM_PATH", "/Users/tmp2/Desktop/iktos/sparsechem"
        ),  # To set this from the CI
        dataloader_num_workers=0,
        imagemodel_hidden_size=[["50"], ["100"]],
        imagemodel_epochs_lr_steps=[
            (5, 3),
        ],
        imagemodel_dropouts=[0.6],
        pseudolabel_threshold=0.95
    )
