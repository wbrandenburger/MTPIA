# ===========================================================================
#   test.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.config.settings
import dl_multi.config.dl_multi
import dl_multi.utils.format

import dl_multi.tasks.tasks

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
get_value = lambda obj, key, default: obj[key] if key in obj.keys() else default

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_default():
    """Default task of set 'test'"""
    dl_multi.__init__._logger.warning("No task chosen from set 'tests'")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_single_task_classification():
    """Call the main routine for evaluation of a single task model"""

    dl_multi.tasks.tasks.task_eval_single_task_classification(setting="test")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_single_task_regression():
    """Call the main routine for evaluation of a single task model"""

    dl_multi.tasks.tasks.task_eval_single_task_regression(setting="test")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_multi_task():
    """Call the main routine for evaluation of a single task model"""

    dl_multi.tasks.tasks.task_eval_multi_task(setting="test")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_tasks():
    """Call the main routine for evaluation of a single task model"""

    dl_multi.tasks.tasks.task_eval_tasks(setting="test")