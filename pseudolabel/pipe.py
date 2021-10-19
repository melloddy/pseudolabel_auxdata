import logging

from pseudolabel import hyperparameters_scan, predict_images_fold2
from pseudolabel.pseudolabel_config import PseudolabelConfig
from pseudolabel.tuner import generation, tuner_tools
from pseudolabel.cp_fitting import splitting_data, fit_cp

LOGGER = logging.getLogger(__name__)


def run_full_pipe(config: PseudolabelConfig):

    LOGGER.info("Starting tuner preprocessing")
    generation.find_overlap_with_melloddy(
        t0_melloddy_path=config.t0_melloddy_path,
        t1_melloddy_path=config.t1_melloddy_path,
        t2_images_path=config.t2_images_path,
        output_folder=config.output_folder_path,
    )
    generation.synchronize_thresholds(
        t8_tunner_path=config.t8c,
        tuner_input_images=config.output_folder_path,
        delete_intermediate_files=False,
    )

    LOGGER.info("Run tuner on images data")
    tuner_tools.run_tuner(
        images_input_folder=config.output_folder_path,
        json_params=config.parameters_json,
        json_key=config.key_json,
        json_ref_hash=config.ref_hash_json,
        output_dir=config.output_folder_path,
        n_cpu=config.max_cpu,
    )
    tuner_tools.process_tuner_output(
        tuner_output_images=config.tuner_output_folder,
        image_features_file=config.t_images_features_path,
    )

    LOGGER.info("Starting hyperparameter scan on image models")
    hyperparameters_scan.run_hyperopt(
        epoch_lr_steps=config.imagemodel_epoch_lr_steps,
        hidden_sizes=config.imagemodel_hidden_size,
        dropouts=config.imagemodel_dropouts,
        hp_output_dir=config.hyperopt_output_folder,
        sparsechem_trainer_path=config.sparsechem_trainer_path,
        tuner_output_dir=config.tuner_output_folder,
        show_progress=config.show_progress,
    )

    LOGGER.info(
        "Selecting best hyperparameter and generating y sparse fold2 for inference"
    )
    best_model = predict_images_fold2.find_best_model(
        hyperopt_folder=config.hyperopt_output_folder
    )
    predict_images_fold2.create_ysparse_fold2(
        tuner_output_images=config.tuner_output_folder,
        intermediate_files_folder=config.intermediate_files_folder,
    )

    LOGGER.info("Predict images fold 2 for fitting conformal predictors")

    predict_images_fold2.run_sparsechem_predict(
        sparsechem_predictor_path=config.sparsechem_predictor_path,
        tuner_output_dir=config.tuner_output_folder,
        best_model=best_model,
        intermediate_files_folder=config.intermediate_files_folder,
        logs_dir=config.log_dir,
    )

    LOGGER.info("Splitting data for conformal predictors training")

    preds_fva, labels, cdvi_fit, cdvi_eval = splitting_data(
        config.tuner_output_folder, config.intermediate_files_folder
    )

    LOGGER.info("Fitting conformal predictors ")

    fit_cp(
        preds_fva=preds_fva,
        labels=labels,
        cdvi_fit=cdvi_fit,
        cdvi_eval=cdvi_eval,
        analysis_folder=config.analysis_folder,
    )
