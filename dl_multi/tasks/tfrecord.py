# ===========================================================================
#   tfrecord.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger
import dl_multi.config.settings
import dl_multi.config.dl_multi
import dl_multi.utils.format
import dl_multi.utils.general as glu
import dl_multi.tftools.tfrecord

import numpy as np
    
#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_default():
    """Default task of set 'test'"""
    _logger.warning("No task chosen from set 'tests'")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_new_tfrecord(setting="training"):
    """Create new tfrecord file"""

    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    try:
        dl_multi.tftools.tfrecord.write_tfrecord(
            dl_multi.config.settings.get_data(setting),
            dl_multi.config.settings._SETTINGS["data-tensor-types"],
            param_label = dl_multi.config.settings._SETTINGS["param_label"],
            param_out = dl_multi.config.settings._SETTINGS["param_out"]
        )
    except KeyError:
        pass

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_print_tfrecord(setting="training"):
    """Create new tfrecord file"""

    try:
        dl_multi.tftools.tfrecord.print_tfrecord(
            dl_multi.config.settings._SETTINGS["data-tensor-types"],
            param_out = dl_multi.config.settings._SETTINGS["param_out"]
        )
    except KeyError:
        pass

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_test_tfrecords_utils():
    """Test the shortcut functions to convert a standard TensorFlow type to a tf.Example-compatible tf.train.Feature
    """

    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.__init__._logger.info("Test the shortcut functions to convert a standard TensorFlow type to a tf.Example-compatible tf.train.Feature")

    # print the results of testing the shortcut functions
    print(dl_multi.tfrecords_utils._bytes_feature(b'test_string'))
    print(dl_multi.tfrecords_utils._bytes_feature(u'test_bytes'.encode('utf-8')))
    print(dl_multi.tfrecords_utils._float_feature(np.exp(1)))
    print(dl_multi.tfrecords_utils._int64_feature(True))
    print(dl_multi.tfrecords_utils._int64_feature(1))
    print(dl_multi.tfrecords_utils._int64_feature(1,True))
