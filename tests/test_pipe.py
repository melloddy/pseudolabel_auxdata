from pseudolabel import pipe
from pseudolabel.pseudolabel_config import PseudolabelConfig


def test_run_full_pipe(config: PseudolabelConfig):
    print(config.__dict__)
    pipe.run_full_pipe(config)
