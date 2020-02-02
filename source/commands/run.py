# ===========================================================================
#   run.py ------------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import __init__
import config.settings
import plugin

import click
import logging
import os
import sys

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
@click.command(
    "run",
    help="Start deep learning framework for xy",
    context_settings=dict(ignore_unknown_options=True)
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
    help="Execute the specified task (default: {0})".format(config.settings._DEFAULT_TASK),
    type=click.Choice([*plugin.get_tasks()]), # @todo[to change]: folder "tasks"
    default=config.settings._DEFAULT_TASK # @todo[to change]: default task "default"
)
@click.option(
    "-f",
    "--func",
    help="Execute the specified function (default: {0})".format(""),
    type=str,
    default= ""
)
def cli(
        file,
        task,
        func,
    ):
    """Read general settings file and execute specified task."""

    # read general settings file and assign content to global settings object
    config.settings.get_settings(file)

    # get the specified task and imort it as module
    task_module = plugin.get_task_module(task)

    # call task's main routine
    if not func:
        __init__._logger.debug("Call the main routine from task module '{0}'".format(task_module[0]))
        task_module[0].main()
    else:
        __init__._logger.debug("Call '{0}' from task module '{1}'".format(task_module[0], func))

        task_funcs = plugin.get_module_functions(task_module[0], "^test")
        if not func in task_funcs:
            raise ValueError("Error: Invalid value for '-f' / '--func': invalid choice: {0}. (choose from {1})".format(
                func, 
                task_funcs
                )
            ) # @todo[generalize]: also in expmgmt

        task_func = getattr(task_module[0], func)
        task_func()