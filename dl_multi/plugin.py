# ===========================================================================
#   plugin.py ---------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger 
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
    _logger.error(exception) # @log

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_module(module):
    module = module if isinstance(module, list) else [module]

    module_name = "dl_multi"
    for sub_module in module:
        module_name = "{0}.{1}".format(module_name, sub_module)

    _logger.debug("Import module '{0}'".format(module_name))

    return (importlib.import_module(module_name), module_name)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_module_from_submodule(module, submodule):
    return get_module([module, submodule])

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_module_task(module, task, submodule=None):
    if submodule is not None:
        module = get_module_from_submodule(module, submodule)[0]

    return getattr(
        module,
        task
    )

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
def get_module_functions(module):
    task_list = list()
    for task in dir(module):
        if re.compile(dl_multi.config.settings._TASK_PREFIX).match(task):
            task_list.append(task.replace(dl_multi.config.settings._TASK_PREFIX ,"",1))
    return task_list
