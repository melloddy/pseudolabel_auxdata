from pseudolabel.pseudolabel_config import PseudolabelConfig
from pseudolabel.predict_all_images import (
    create_x_ysparse_all_images,
    run_sparsechem_predict,
)
from pseudolabel.predict_images_fold2 import find_best_model


def test_create_x_ysparse_all_images(config: PseudolabelConfig):
    create_x_ysparse_all_images(
        tuner_output_image=config.tuner_output_folder,
        t_images_features_path=config.t_images_features_path,
        analysis_folder=config.analysis_folder,
        intermediate_files_folder=config.intermediate_files_folder,
    )


def test_run_sparsechem_predict(config: PseudolabelConfig):
    best_model = find_best_model(hyperopt_folder=config.hyperopt_output_folder)

    run_sparsechem_predict(
        sparsechem_predictor_path=config.sparsechem_predictor_path,
        best_model=best_model,
        intermediate_files_folder=config.intermediate_files_folder,
        logs_dir=config.log_dir,
        torch_device=config.torch_device,
        dataloader_num_workers=config.dataloader_num_workers,
    )
