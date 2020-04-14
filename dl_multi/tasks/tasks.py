# ===========================================================================
#   tasks.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger
import dl_multi.config.settings
import dl_multi.config.dl_multi
import dl_multi.utils.general as glu
import dl_multi.utils.format

import dl_multi.models.train_multi_task
import dl_multi.models.train_single_task_classification
import dl_multi.models.train_single_task_regression

import dl_multi.eval.eval_tasks
import dl_multi.eval.eval_multi_task
import dl_multi.eval.eval_single_task_classification
import dl_multi.eval.eval_single_task_regression

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_default():
    """Default task of set 'test'"""
    _logger.warning("No task chosen from set 'tests'")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_train_single_task_classification():
    """Call the main routine for training of a single task classification model"""
    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.models.train_single_task_classification.train(
        dl_multi.config.settings._SETTINGS["param_log"],
        dl_multi.config.settings._SETTINGS["param_batch"],
        dl_multi.config.settings._SETTINGS["param_save"],      
        dl_multi.config.settings._SETTINGS["param_train"]
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_train_single_task_regression():
    """Call the main routine for training of a single task regression model"""
    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.models.train_single_task_regression.train(
        dl_multi.config.settings._SETTINGS["param_log"],
        dl_multi.config.settings._SETTINGS["param_batch"],
        dl_multi.config.settings._SETTINGS["param_save"],      
        dl_multi.config.settings._SETTINGS["param_train"]
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_train_multi_task():
    """Call the main routine for training of a multi task model"""
    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.models.train_multi_task.train(
        dl_multi.config.settings._SETTINGS["param_specs"],
        dl_multi.config.settings._SETTINGS["param_info"],
        dl_multi.config.settings._SETTINGS["param_log"],
        dl_multi.config.settings._SETTINGS["param_batch"],
        dl_multi.config.settings._SETTINGS["param_save"],      
        dl_multi.config.settings._SETTINGS["param_train"]
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_single_task_classification(setting="training"):
    """Call the main routine for evaluation of a single task classifcation model with training data"""
    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    ) 

    dl_multi.eval.eval_single_task_classification.eval(
        dl_multi.config.settings.get_data(setting),
        dl_multi.config.settings._SETTINGS["param_specs"],
        dl_multi.config.settings._SETTINGS["param_io"],
        dl_multi.config.settings._SETTINGS["param_log"],      
        dl_multi.config.settings._SETTINGS["param_eval"],
        dl_multi.config.settings._SETTINGS["param_label"],
        dl_multi.config.settings._SETTINGS["param_class"]
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_single_task_regression(setting="training"):
    """Call the main routine for evaluation of a single task regression model with training data"""
    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.eval.eval_single_task_regression.eval(
        dl_multi.config.settings.get_data(setting),
        dl_multi.config.settings._SETTINGS["param_specs"],
        dl_multi.config.settings._SETTINGS["param_io"],
        dl_multi.config.settings._SETTINGS["param_log"],         
        dl_multi.config.settings._SETTINGS["param_eval"],
        dl_multi.config.settings._SETTINGS["param_label"],
        dl_multi.config.settings._SETTINGS["param_class"]
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_multi_task(setting="training"):
    """Call the main routine for evaluation of a multi task model with training data"""
    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.eval.eval_multi_task.eval(
        dl_multi.config.settings.get_data(setting),
        dl_multi.config.settings._SETTINGS["param_specs"],
        dl_multi.config.settings._SETTINGS["param_io"],
        dl_multi.config.settings._SETTINGS["param_log"],      
        dl_multi.config.settings._SETTINGS["param_eval"],
        dl_multi.config.settings._SETTINGS["param_label"],
        dl_multi.config.settings._SETTINGS["param_class"]
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_eval_tasks(setting="training"):
    """Call the main routine for the generalized evaluation of a single or multi task model with training data"""
    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.eval.eval_tasks.eval(
        dl_multi.config.settings.get_data(setting),
        dl_multi.config.settings._SETTINGS["param_specs"],
        dl_multi.config.settings._SETTINGS["param_io"],
        dl_multi.config.settings._SETTINGS["param_log"],                
        dl_multi.config.settings._SETTINGS["param_eval"],
        dl_multi.config.settings._SETTINGS["param_label"],
        dl_multi.config.settings._SETTINGS["param_class"]
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_print_user_settings():
    """Print the user settings"""
    
    # print user's defined settings
    _logger.info("Print user's defined settings")
    dl_multi.utils.format.print_data(dl_multi.config.settings._SETTINGS)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_print_user_data():
    """Print the user data"""
    
    # print user's defined data
    _logger.info("Print user's defined data")
    dl_multi.utils.format.print_data(dl_multi.config.settings.get_data_dict())