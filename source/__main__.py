# ===========================================================================
#   main.py -----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import __init__
import config.settings

import click
import logging
import importlib
import os
import sys

#   settings ----------------------------------------------------------------
# ---------------------------------------------------------------------------
_TASK_DIR = "tasks"
_DEFAULT_TASK = "default"

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_main_task(path):
    
    file_list = [os.path.splitext(f)[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if file_list == list():
        raise ValueError("The predefined task folder seems to be empty.")

    return file_list

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
    help="Execute the specified task (default: {0})".format(_DEFAULT_TASK),
    type=click.Choice([*get_main_task(os.path.join(os.path.dirname(__file__), _TASK_DIR))]), # @todo[to change]: folder "tasks"
    default=_DEFAULT_TASK # @todo[to change]: default task "default"
)
@click.option(
    "-f",
    "--func",
    help="Execute the specified function (default: {0})".format(""),
    type=str,
    default= ""
)
@click.option(
    "-d",
    "--data",
    help="Pass the trainings, test and validation of the specified setting.",
    nargs=2, 
    type=(str, str)
)
def cli(
        file,
        task,
        func,
        data
    ):
    """Read general settings file and execute specified task."""

    # read general settings file and assign content to global settings object
    config.settings.get_settings(file)
 
    print(data)
    # get the specified task and imort it as module
    get_main_task(os.path.join(os.path.dirname(__file__), _TASK_DIR))

    module_string = "{0}.{1}".format(_TASK_DIR, task)
    __init__._logger.debug("Import task module '{0}'".format(module_string))
    task_module = importlib.import_module(module_string)

    # call task's main routine
    if not func:
        __init__._logger.debug("Call the main routine from task module '{0}'".format(module_string))
        task_module.main()
    else:
        __init__._logger.debug("Call '{0}' from task module '{1}'".format(module_string, func))
        task_func = getattr(task_module, func)
        task_func()

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Main"""
    
    # log all retrieved arguments
    __init__._logger.debug("Number of arguments {0}:".format(len(sys.argv)))
    __init__._logger.debug("CLI-Arguments are: {0}".format(str(sys.argv)))

    # call default command line interface
    cli()
