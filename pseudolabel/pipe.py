import logging
import os
from typing import Optional

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


class PseudolabelPipe:
    def __init__(self, config_file: str = None):
        self.config: Optional[PseudolabelConfig] = None
        self.steps_list = [
            "data_preprocessing",
            "image_model_hyperscan",
            "fit_conformal_predictors",
            "generate_pseudolabels",
            "generate_T_files_pseudolabels",
        ]
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: str):

        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file is not found : {config_file}")

        self.config = PseudolabelConfig.load_config(config_file)

    def run_partial_pipe(self, starting_step: str):
        if starting_step not in self.steps_list:
            raise ValueError(
                f"{starting_step} is not a valid starting step, please choose one on the following : "
                f"{self.steps_list}"
            )
        starting_step_ind = self.steps_list.index(starting_step)
        self.run_full_pipe(starting_step_ind)

    def run_full_pipe(self, starting_ind=0):
        # TODO Add step number/steps
        if starting_ind <= 0:
            LOGGER.info("STEP 1/5: Starting tuner preprocessing")
            generation.find_overlap_with_melloddy(
                t0_melloddy_path=self.config.t0_melloddy_path,
                t1_melloddy_path=self.config.t1_melloddy_path,
                t2_images_path=self.config.t2_images_path,
                t_catalogue_path=self.config.t_catalogue_path,
                tuner_images_input_folder=self.config.tuner_input_folder_image,
            )
            generation.synchronize_thresholds(
                t8_tunner_path=self.config.t8c_baseline,
                tuner_images_input_folder=self.config.tuner_input_folder_image,
                delete_intermediate_files=False,
            )

            LOGGER.info("Run tuner on images data")
            tuner_tools.run_tuner(
                tuner_images_input_folder=self.config.tuner_input_folder_image,
                json_params=self.config.parameters_json,
                json_key=self.config.key_json,
                json_ref_hash=self.config.ref_hash_json,
                output_dir=os.path.join(self.config.tuner_output_folder_image, ".."),
                n_cpu=self.config.max_cpu,
            )
            tuner_tools.process_tuner_output(
                tuner_output_images=self.config.tuner_output_folder_image,
                image_features_file=self.config.t_images_features_path,
            )

        if starting_ind <= 1:
            LOGGER.info("STEP 2/5: Starting hyperparameter scan on image models")
            hyperparameters_scan.run_hyperopt(
                epochs_lr_steps=self.config.imagemodel_epochs_lr_steps,
                hidden_sizes=self.config.imagemodel_hidden_size,
                dropouts=self.config.imagemodel_dropouts,
                hp_output_dir=self.config.hyperopt_output_folder,
                sparsechem_trainer_path=self.config.sparsechem_trainer_path,
                tuner_output_dir=self.config.tuner_output_folder_image,
                validation_fold=self.config.validation_fold,
                test_fold=self.config.test_fold,
                show_progress=self.config.show_progress,
                resume_hyperopt=self.config.resume_hyperopt,
                torch_device=self.config.torch_device,
                hyperopt_subset_ind=self.config.hyperopt_subset_ind,
            )

        if starting_ind <= 2:
            LOGGER.info(
                "STEP 3/5: Selecting best hyperparameter and generating y sparse fold val for inference"
            )
            if not hyperparameters_scan.hyperopt_completion_status(
                hyperopt_size=self.config.hyperopt_size,
                hp_output_dir=self.config.hyperopt_output_folder,
            ):
                LOGGER.info("Hyperopt not complete yet - stopping here.")
                quit()
            best_model = predict_images_fold_val.find_best_model(
                hyperopt_folder=self.config.hyperopt_output_folder
            )
            predict_images_fold_val.create_ysparse_fold_val(
                tuner_output_images=self.config.tuner_output_folder_image,
                intermediate_files_folder=self.config.intermediate_files_folder,
                validation_fold=self.config.validation_fold,
            )

            LOGGER.info("Predict images fold val for fitting conformal predictors")

            predict_images_fold_val.run_sparsechem_predict(
                sparsechem_predictor_path=self.config.sparsechem_predictor_path,
                tuner_output_dir=self.config.tuner_output_folder_image,
                best_model=best_model,
                intermediate_files_folder=self.config.intermediate_files_folder,
                logs_dir=self.config.log_dir,
                dataloader_num_workers=self.config.dataloader_num_workers,
                torch_device=self.config.torch_device,
            )

            LOGGER.info("Splitting data for conformal predictors training")

            preds_fva, labels, cdvi_fit, cdvi_eval = cp_fitting.splitting_data(
                tuner_output_images=self.config.tuner_output_folder_image,
                intermediate_files_folder=self.config.intermediate_files_folder,
                fold_va=self.config.validation_fold,
            )

            LOGGER.info("Fitting conformal predictors ")

            cp_fitting.fit_cp(
                preds=preds_fva,
                labels=labels,
                cdvi_fit=cdvi_fit,
                cdvi_eval=cdvi_eval,
                analysis_folder=self.config.analysis_folder,
            )

            LOGGER.info("Saving task stats on pseudolabels")

            cp_fitting.generate_task_stats(analysis_folder=self.config.analysis_folder)

        if starting_ind <= 3:
            LOGGER.info(
                "STEP 4/5: Generating x and y sparse for all images on selected tasks for inference"
            )
            predict_all_images.create_x_ysparse_all_images(
                tuner_output_image=self.config.tuner_output_folder_image,
                t_images_features_path=self.config.t_images_features_path,
                analysis_folder=self.config.analysis_folder,
                intermediate_files_folder=self.config.intermediate_files_folder,
                size_batch=self.config.x_ysparse_batch_size,
            )

            LOGGER.info("Generating predictions on all image compounds")
            if not hyperparameters_scan.hyperopt_completion_status(
                hyperopt_size=self.config.hyperopt_size,
                hp_output_dir=self.config.hyperopt_output_folder,
            ):
                LOGGER.info("Hyperopt not complete yet - stopping here.")
                quit()
            best_model = predict_images_fold_val.find_best_model(
                hyperopt_folder=self.config.hyperopt_output_folder
            )
            predict_all_images.run_sparsechem_predict(
                sparsechem_predictor_path=self.config.sparsechem_predictor_path,
                best_model=best_model,
                intermediate_files_folder=self.config.intermediate_files_folder,
                logs_dir=self.config.log_dir,
                dataloader_num_workers=self.config.dataloader_num_workers,
                torch_device=self.config.torch_device,
            )

            LOGGER.info("Apply conformal predictors on all images predictions")

            cp_fitting.apply_cp_aux(
                analysis_folder=self.config.analysis_folder,
                t2_images_path=self.config.t2_images_path,
                intermediate_files=self.config.intermediate_files_folder,
            )

        if starting_ind <= 4:
            LOGGER.info("STEP 5/5: Creating T files for pseudolabels auxiliary tasks")

            t_pseudolabels_generation.generate_t_aux_pl(
                intermediate_files_folder=self.config.intermediate_files_folder,
                t2_melloddy_path=self.config.t2_melloddy_path,
                t2_images_path=self.config.t2_images_path,
            )

            LOGGER.info("Replacing pseudolabels values with true labels if applicable")

            t_pseudolabels_generation.find_labels_of_auxtasks(
                tuner_output_folder_baseline=self.config.tuner_output_folder_baseline,
                tuner_output_folder_image=self.config.tuner_output_folder_image,
                intermediate_files_folder=self.config.intermediate_files_folder,
            )
            t_pseudolabels_generation.replace_pseudolabels_w_labels(
                intermediate_files_folder=self.config.intermediate_files_folder,
            )

            t_pseudolabels_generation.filter_low_confidence_pseudolabels(
                intermediate_files_folder=self.config.intermediate_files_folder,
                output_pseudolabel_folder=self.config.output_pseudolabel_folder,
                analysis_folder=self.config.analysis_folder,
                threshold=self.config.pseudolabel_threshold,
            )

            LOGGER.info("T files for pseudolabels are generated")
        LOGGER.info("Pipeline finished")
