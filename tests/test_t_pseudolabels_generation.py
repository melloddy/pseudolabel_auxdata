from pseudolabel.t_pseudolabels_generation import generate_t_aux_pl


def test_generate_t_aux_pl(config):
    generate_t_aux_pl(
        intermediate_files_folder=config.intermediate_files_folder,
        t0_melloddy_path=config.t0_melloddy_path,
        t2_images_path=config.t2_images_path,
    )
