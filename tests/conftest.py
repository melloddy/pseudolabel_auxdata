from pytest import fixture


@fixture(scope="session")
def images_folder():
    return "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/output"


@fixture(scope="session")
def params_folder():
    return "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/params"


@fixture(scope="session")
def tuner_images_output_folder():
    return (
        "/home/robin/dev/iktos/mellody/wp1/pseudolabel_auxdata/data/output/tuner_images"
    )
