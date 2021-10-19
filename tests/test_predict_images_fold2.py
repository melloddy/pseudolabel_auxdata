import os.path

from pseudolabel import predict_images_fold2
from pseudolabel.constants import IMAGE_MODEL_NAME


def test_find_best_model(config):
    best_model = predict_images_fold2.find_best_model(
        hyperopt_folder=config.hyperopt_output_folder,
    )

    assert os.path.exists(os.path.join(best_model, f"{IMAGE_MODEL_NAME}.json"))


def test_create_ysparse_fold2(config):
    predict_images_fold2.create_ysparse_fold2(
        tuner_output_images=config.tuner_output_folder,
        intermediate_files_folder=config.intermediate_files_folder,
    )

    assert os.path.exists(
        os.path.join(config.intermediate_files_folder, "main_tasks_with_image.npy")
    )
    assert os.path.exists(
        os.path.join(
            config.intermediate_files_folder, "y_sparse_step1_main_tasks_fold2.npy"
        )
    )


def test_find_best_model_and_run_sparsechem(config):
    best_model = predict_images_fold2.find_best_model(
        hyperopt_folder=config.hyperopt_output_folder,
    )

    predict_images_fold2.run_sparsechem_predict(
        sparsechem_predictor_path=config.sparsechem_predictor_path,
        tuner_output_dir=config.tuner_output_folder,
        best_model=best_model,
        intermediate_files_folder=config.intermediate_files_folder,
        logs_dir=config.log_dir,
    )
