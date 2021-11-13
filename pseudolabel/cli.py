import logging
import sys

import click

from pseudolabel import pipe

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s",
)

logging.getLogger("matplotlib.font_manager").disabled = True


@click.command()
@click.option("-c", "--config-file", required=True, type=str)
def run_pipe(config_file: str):
    pseudolabel_pipe = pipe.PseudolabelPipe(config_file)
    pseudolabel_pipe.run_full_pipe()


@click.command()
@click.option("-c", "--config-file", required=True, type=str)
@click.option("-s", "--starting_step", required=True, type=str)
def run_parital_pipe(config_file: str, starting_step: str):
    pseudolabel_pipe = pipe.PseudolabelPipe(config_file)
    pseudolabel_pipe.run_partial_pipe(starting_step=starting_step)