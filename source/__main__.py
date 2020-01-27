# ===========================================================================
#   main.py -----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import __init__
import hello_train
import config.settings

import sys
import click
import logging

#   settings ----------------------------------------------------------------
# ---------------------------------------------------------------------------
logger = logging.getLogger("main")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_main_task():
    return ["hello", "data"]

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
@click.command(
    help="Start deep learning framework for xy"
)
@click.help_option(
    "-h",
    "--help" 
)
@click.argument(
    "file", 
    type=str, 
    nargs=1
)
@click.option(
    "-t",
    "--task",
    help="Execute the specified task (default:{0})".format(get_main_task()[0]),
    type=click.Choice([*get_main_task()]),
    default=get_main_task()[0]
)
def cli(
    file,
    task
    ):
    """Read general settings file and execute specified task."""

    # read general settings file and assign content to global settings object
    config.settings.get_settings(file)

    print(config.settings._SETTINGS)


#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Main"""
    
    # log all retrieved arguments
    logger.debug("Number of arguments {0}:".format(len(sys.argv)))
    logger.debug("CLI-Arguments are: {0}".format(str(sys.argv)))

    # call default command line interface
    cli()
