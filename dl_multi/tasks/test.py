# ===========================================================================
#   test.py -----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.config.settings
import dl_multi.config.dl_multi
import dl_multi.utils.format

import os

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
get_value = lambda obj, key, default: obj[key] if key in obj.keys() else default

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_default():
    """Default task of set 'test'"""
    dl_multi.__init__._logger.warning("No task chosen from set 'tests'")