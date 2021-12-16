import logging
import os

from pseudolabel import (
    cp_fitting,
    hyperparameters_scan,
    predict_all_images,
    predict_images_fold_val,
    t_pseudolabels_generation,
)
from pseudolabel.pseudolabel_config import PseudolabelConfig
from pseudolabel.tuner import generation, tuner_tools

LOGGER = logging.getLogger(__name__)


def data_preprocessing_step(config: PseudolabelConfig):
    LOGGER.info("STEP 1/6: Starting tuner preprocessing")
    generation.find_overlap_with_melloddy(
        t0_melloddy_path=config.t0_melloddy_path,
        t1_melloddy_path=config.t1_melloddy_path,
        t2_images_path=config.t2_images_path,
        t_catalogue_path=config.t_catalogue_path,
        tuner_images_input_folder=config.tuner_input_folder_image,
    )
    generation.synchronize_thresholds(
        t8_tunner_path=config.t8c_baseline,
        tuner_images_input_folder=config.tuner_input_folder_image,
        delete_intermediate_files=False,
    )

    LOGGER.info("Run tuner on images data")
    tuner_tools.run_tuner(
        tuner_images_input_folder=config.tuner_input_folder_image,
        json_params=config.parameters_json,
        json_key=config.key_json,
        json_ref_hash=config.ref_hash_json,
        output_dir=os.path.join(config.tuner_output_folder_image, ".."),
        n_cpu=config.max_cpu,
    )
    tuner_tools.process_tuner_output(
        tuner_output_images=config.tuner_output_folder_image,
        image_features_file=config.t_images_features_path,
    )


def image_model_hyperscan_step(config: PseudolabelConfig):
    LOGGER.info("STEP 2/6: Starting hyperparameter scan on image models")
    hyperparameters_scan.run_hyperopt(
        epochs_lr_steps=config.imagemodel_epochs_lr_steps,
        hidden_sizes=config.imagemodel_hidden_size,
        dropouts=config.imagemodel_dropouts,
        hp_output_dir=config.hyperopt_output_folder,
        sparsechem_trainer_path=config.sparsechem_trainer_path,
        tuner_output_dir=config.tuner_output_folder_image,
        validation_fold=config.validation_fold,
        test_fold=config.test_fold,
        show_progress=config.show_progress,
        resume_hyperopt=config.resume_hyperopt,
        torch_device=config.torch_device,
        hyperopt_subset_ind=config.hyperopt_subset_ind,
    )


def fitting_cp_step(config: PseudolabelConfig):
    LOGGER.info(
        "STEP 3/6: Selecting best hyperparameter and generating y sparse fold val for inference"
    )
    if not hyperparameters_scan.hyperopt_completion_status(
        hyperopt_size=config.hyperopt_size,
        hp_output_dir=config.hyperopt_output_folder,
    ):
        LOGGER.info("Hyperopt not complete yet - stopping here.")
        quit()
    best_model = predict_images_fold_val.find_best_model(
        hyperopt_folder=config.hyperopt_output_folder
    )
    predict_images_fold_val.create_ysparse_fold_val(
        tuner_output_images=config.tuner_output_folder_image,
        intermediate_files_folder=config.intermediate_files_folder,
        validation_fold=config.validation_fold,
    )

    LOGGER.info("Predict images fold val for fitting conformal predictors")

    predict_images_fold_val.run_sparsechem_predict(
        sparsechem_predictor_path=config.sparsechem_predictor_path,
        tuner_output_dir=config.tuner_output_folder_image,
        best_model=best_model,
        intermediate_files_folder=config.intermediate_files_folder,
        logs_dir=config.log_dir,
        dataloader_num_workers=config.dataloader_num_workers,
        torch_device=config.torch_device,
    )

    LOGGER.info("Splitting data for conformal predictors training")

    preds_fva, labels, cdvi_fit, cdvi_eval = cp_fitting.splitting_data(
        tuner_output_images=config.tuner_output_folder_image,
        intermediate_files_folder=config.intermediate_files_folder,
        fold_va=config.validation_fold,
    )

    LOGGER.info("Fitting conformal predictors ")

    cp_fitting.fit_cp(
        preds=preds_fva,
        labels=labels,
        cdvi_fit=cdvi_fit,
        cdvi_eval=cdvi_eval,
        analysis_folder=config.analysis_folder,
    )

    LOGGER.info("Saving task stats on pseudolabels")

    cp_fitting.generate_task_stats(analysis_folder=config.analysis_folder)


def generate_all_pred_step(config: PseudolabelConfig):
    LOGGER.info(
        "STEP 4/6: Generating x and y sparse for all images on selected tasks for inference"
    )
    predict_all_images.create_x_ysparse_all_images(
        tuner_output_image=config.tuner_output_folder_image,
        t_images_features_path=config.t_images_features_path,
        analysis_folder=config.analysis_folder,
        intermediate_files_folder=config.intermediate_files_folder,
        size_batch=config.x_ysparse_batch_size,
    )

    LOGGER.info("Generating predictions on all image compounds")
    if not hyperparameters_scan.hyperopt_completion_status(
        hyperopt_size=config.hyperopt_size,
        hp_output_dir=config.hyperopt_output_folder,
    ):
        LOGGER.info("Hyperopt not complete yet - stopping here.")
        quit()
    best_model = predict_images_fold_val.find_best_model(
        hyperopt_folder=config.hyperopt_output_folder
    )
    predict_all_images.run_sparsechem_predict(
        sparsechem_predictor_path=config.sparsechem_predictor_path,
        best_model=best_model,
        intermediate_files_folder=config.intermediate_files_folder,
        logs_dir=config.log_dir,
        dataloader_num_workers=config.dataloader_num_workers,
        torch_device=config.torch_device,
    )
    if config.number_task_batches > 0:
        LOGGER.info(
            "batch apply-cp-aux mode detected. Stopping here for parallel apply-cp-aux submission"
        )
        quit()


def apply_cp_step(config: PseudolabelConfig):
    LOGGER.info("STEP 5/6 : Apply conformal predictors on all images predictions")

    cp_fitting.apply_cp_aux(
        analysis_folder=config.analysis_folder,
        t2_images_path=config.t2_images_path,
        intermediate_files=config.intermediate_files_folder,
        num_task_batch=config.number_task_batches,
        task_batch=config.apply_cp_to_task_batch,
    )
    if config.number_task_batches > 0:
        # needs to quit here and make sure we have all batch runs before starting next step.
        # we need to stop here and check manually
        # unless we have a way to find out - at this point - the expected number of tasks
        # that have data saved in files produced by apply_cp_aux, we cannot proceed to next step
        LOGGER.info(
            "Done, please verify the completion - and restart pipeline from next step once all batches are complete (generate_T_files_pseudolabels)"
        )
        quit()


def generate_T_files_pl_step(config: PseudolabelConfig):
    LOGGER.info("STEP 6/6: Creating T files for pseudolabels auxiliary tasks")

    t_pseudolabels_generation.generate_t_aux_pl(
        intermediate_files_folder=config.intermediate_files_folder,
        t2_melloddy_path=config.t2_melloddy_path,
        t2_images_path=config.t2_images_path,
        num_task_batch=config.number_task_batches,
        assay_id_offset=config.pseudolabel_assay_id_offset,
    )

    LOGGER.info("Replacing pseudolabels values with true labels if applicable")

    t_pseudolabels_generation.find_labels_of_auxtasks(
        tuner_output_folder_baseline=config.tuner_output_folder_baseline,
        tuner_output_folder_image=config.tuner_output_folder_image,
        intermediate_files_folder=config.intermediate_files_folder,
    )
    t_pseudolabels_generation.replace_pseudolabels_w_labels(
        intermediate_files_folder=config.intermediate_files_folder,
    )

    t_pseudolabels_generation.filter_low_confidence_pseudolabels(
        intermediate_files_folder=config.intermediate_files_folder,
        output_pseudolabel_folder=config.output_pseudolabel_folder,
        analysis_folder=config.analysis_folder,
        threshold=config.pseudolabel_threshold,
    )

    LOGGER.info("T files for pseudolabels are generated")
