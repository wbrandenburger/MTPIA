# ===========================================================================
#   test.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger
import dl_multi.config.settings
import dl_multi.config.dl_multi
import dl_multi.utils.format

import dl_multi.tasks.tasks

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_default():
    """Default task of set 'test'"""
    _logger.warning("No task chosen from set 'tests'")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_single_task_classification():
    """Call the main routine for evaluation of a single task classification model"""

    dl_multi.tasks.tasks.task_eval_single_task_classification(setting="test")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_single_task_regression():
    """Call the main routine for evaluation of a single task regression model with test data"""

    dl_multi.tasks.tasks.task_eval_single_task_regression(setting="test")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_multi_task():
    """Call the main routine for evaluation of a multi task model with test data"""

    dl_multi.tasks.tasks.task_eval_multi_task(setting="test")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_tasks():
    """Call the main routine for the generalized evaluation of a single or multi task model with test data"""

    dl_multi.tasks.tasks.task_eval_tasks(setting="test")