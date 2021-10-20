from pseudolabel.t_pseudolabels_generation import (
    generate_t_aux_pl,
    find_labels_of_auxtasks,
    replace_pseudolabels_w_labels,
)


def test_generate_t_aux_pl(config):
    generate_t_aux_pl(
        intermediate_files_folder=config.intermediate_files_folder,
        t0_melloddy_path=config.t0_melloddy_path,
        t2_images_path=config.t2_images_path,
    )


def test_find_labels_of_auxtasks(config):
    find_labels_of_auxtasks(
        tuner_output_folder_baseline=config.tuner_output_folder_baseline,
        tuner_output_folder_image=config.tuner_output_folder_image,
        intermediate_files_folder=config.intermediate_files_folder,
    )


def test_replace_pseudolabels_w_labels(config):
    replace_pseudolabels_w_labels(
        intermediate_files_folder=config.intermediate_files_folder,
        output_pseudolabel_folder=config.output_pseudolabel_folder,
    )
