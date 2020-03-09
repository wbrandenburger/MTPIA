# ===========================================================================
#   tasks.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.config.settings
import dl_multi.config.dl_multi
import dl_multi.utils.format
import dl_multi.tools.train_multi
import dl_multi.tools.train_sem
import dl_multi.tools.train_dsm
import dl_multi.tools.eval_multi
import dl_multi.tools.eval_sem
import dl_multi.tools.eval_dsm

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
get_value = lambda obj, key, default: obj[key] if key in obj.keys() else default

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_default():
    """Default task of set 'tasks'"""
    task_print_user_settings()

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_multi():
    """Call the main routine for evaluation of a single task model"""
    
    dl_multi.tools.eval_multi.eval(
        get_value(dl_multi.config.settings._SETTINGS, "param_eval", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_label", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_color", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_out", dict())
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_train_multi():
    """Call the main routine for training a single task model"""
    dl_multi.config.dl_multi.set_cuda_properties(
        get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.tools.train_multi.train(
        get_value(dl_multi.config.settings._SETTINGS, "param_train", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_out", dict())
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_train_sem():
    """Call the main routine for training a single task model"""
    dl_multi.config.dl_multi.set_cuda_properties(
        get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.tools.train_sem.train(
        get_value(dl_multi.config.settings._SETTINGS, "param_train", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_out", dict())
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_sem():
    """Call the main routine for evaluation of a single task model"""
    
    dl_multi.tools.eval_sem.eval(
        get_value(dl_multi.config.settings._SETTINGS, "param_eval", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_label", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_color", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_out", dict())
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_train_dsm():
    """Call the main routine for training a single task model"""
    dl_multi.config.dl_multi.set_cuda_properties(
        get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.tools.train_dsm.train(
        get_value(dl_multi.config.settings._SETTINGS, "param_train", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_out", dict())
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_dsm():
    """Call the main routine for evaluation of a single task model"""
    
    dl_multi.tools.eval_dsm.eval(
        get_value(dl_multi.config.settings._SETTINGS, "param_eval", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_label", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_color", dict()),
        get_value(dl_multi.config.settings._SETTINGS, "param_out", dict())
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
    dl_multi.utils.format.print_data(dl_multi.config.settings.get_data_dict())