from pseudolabel import hyperparameters_scan


def test_run_hyperopt(config):
    hyperparameters_scan.run_hyperopt(
        epoch_lr_steps=config.imagemodel_epoch_lr_steps,
        hidden_sizes=config.imagemodel_hidden_size,
        dropouts=config.imagemodel_dropouts,
        hp_output_dir=config.hyperopt_output_folder,
        sparsechem_trainer_path=config.sparsechem_trainer_path,
        tuner_output_dir=config.tuner_output_folder,
        show_progress=config.show_progress,
    )
