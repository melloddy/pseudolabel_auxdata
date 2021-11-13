from pseudolabel import pipe
from pseudolabel.pseudolabel_config import PseudolabelConfig


def test_run_full_pipe(config: PseudolabelConfig):
    print(str(config.__dict__).replace(',', '\n'))
    pseudolabel_pip = pipe.PseudolabelPipe()
    pseudolabel_pip.config = config
    pseudolabel_pip.run_full_pipe()


def test_run_partial_pip(config: PseudolabelConfig):
    print(str(config.__dict__).replace(',', '\n'))
    pseudolabel_pip = pipe.PseudolabelPipe()
    pseudolabel_pip.config = config
    pseudolabel_pip.run_partial_pipe("generate_pseudolabels")
