# ===========================================================================
#   run.py ------------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.config.settings
import dl_multi.plugin
import dl_multi.debug.exceptions

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
    "--task_set",
    help="Execute a task from specified task set(default: {0})".format(dl_multi.config.settings._TASK_SPEC_NAME),
    type=click.Choice([*dl_multi.plugin.get_tasks()]),
    default=dl_multi.config.settings._TASK_SPEC_NAME
)
@click.option(
    "-t",
    "--task",
    help="Execute the specified task (default: {0})".format(""),
    type=str,
    default= dl_multi.config.settings._DEFAULT_TASK 
)
def cli(
        file,
        task,
        func,
    ):
    """Read general settings file and execute specified task."""

    # read general settings file and assign content to global settings object
    dl_multi.config.settings.get_settings(file)

    # get the specified task and imort it as module
    task_module = dl_multi.plugin.get_task_module(task)

    # call task's main routine
    if not task:
        dl_multi.__init__._logger.debug("Call the default routine from task set '{0}'".format(task_module[0]))
        task_module[0].main()
    else:
        dl_multi.__init__._logger.debug("Call task '{0}' from set '{1}'".format(task_module[0], task))

        task_funcs = dl_multi.plugin.get_module_functions(task_module[0])
        if not task in task_funcs:
            raise dl_multi.debug.exceptions.ArgumentError(task, task_funcs) 

        task_func = getattr(task_module[0], 
            "{}{}".format(dl_multi.config.settings._TASK_PREFIX, task)
        )
        task_func()