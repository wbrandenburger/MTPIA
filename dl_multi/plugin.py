# ===========================================================================
#   plugin.py ---------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.config.settings

import importlib
import logging
import os
import re

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def stevedore_error_handler(manager, entrypoint, exception):
    dl_multi.__init__._logger.error(
        "Error while loading entrypoint [{0}]".format(entrypoint)
    ) # @log
    dl_multi.__init__._logger.error(exception) # @log

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_tasks():
    module = importlib.import_module("dl_multi.{0}".format(dl_multi.config.settings._TASK_DIR))
    path = os.path.dirname(module.__file__)
    file_list = [os.path.splitext(f)[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and re.compile("[^__.+__$]").match(f)]

    if file_list == list():
        raise ValueError("The predefined task folder seems to be empty.")

    return file_list

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_task_module(task):
    module_name = "dl_multi.{0}.{1}".format(dl_multi.config.settings._TASK_DIR, task)
    dl_multi.__init__._logger.debug("Import task module '{0}'".format(module_name))
    
    return (importlib.import_module(module_name), module_name)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_module_functions(module):
    task_list = list()
    for task in dir(module):
        if re.compile(dl_multi.config.settings._TASK_PREFIX).match(task):
            task_list.append(task.replace(dl_multi.config.settings._TASK_PREFIX ,"",1))
    return task_list
