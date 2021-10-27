import logging
import sys

import click

from pseudolabel import PseudolabelConfig, pipe

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s",
)


@click.command()
@click.option("-c", "--config-file", required=True, type=str)
def run_pipe(config_file: str):
    config = PseudolabelConfig.load_config(config_file)
    pipe.run_full_pipe(config)
