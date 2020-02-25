# ===========================================================================
#   tasks.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.config.settings
import dl_multi.config.dl_multi
import dl_multi.utils.format

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
get_value = lambda obj, key, default: obj[key] if key in obj.keys() else default

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_default():
    task_print_user_settings()

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_train():
    dl_multi.config.settings.set_cuda_properties(
        get_value(dl_multi.config.settings._SETTINGS["param_cuda"], dict())
    )  

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_print_user_settings():
    """Print the user settings"""
    
    # print user's defined settings
    dl_multi.__init__._logger.info("Print user's defined settings")
    dl_multi.utils.format.print_data(dl_multi.config.settings._SETTINGS)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_print_user_data():
    """Print the user data"""
    
    # print user's defined data
    dl_multi.__init__._logger.info("Print user's defined data")
    dl_multi.utils.format.print_data(dl_multi.config.settings._DATA)