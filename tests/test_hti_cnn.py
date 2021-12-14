from pseudolabel import PseudolabelConfig
from pseudolabel import pipe


def test_run_hti(config: PseudolabelConfig):
    print(str(config.__dict__).replace(',', '\n'))
    pseudolabel_pip = pipe.PseudolabelPipe()
    pseudolabel_pip.config = config
    pseudolabel_pip.run_hti_cnn()
