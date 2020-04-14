# ===========================================================================
#   tfrecord.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger
import dl_multi.config.settings
import dl_multi.config.dl_multi
import dl_multi.tftools.tfutils
import dl_multi.utils.format
import dl_multi.utils.general as glu

import dl_multi.tftools.tfrecord

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_default():
    """Default task of set 'test'"""
    _logger.warning("No task chosen from set 'tests'")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_new_tfrecord_file(setting="training"):
    """Create a new tfrecord file."""

    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.tftools.tfrecord.write_tfrecord(
        dl_multi.config.settings.get_data(setting),
        dl_multi.config.settings._SETTINGS["param_specs"],
        dl_multi.config.settings._SETTINGS["param_tfrecord"],
        param_label = glu.get_value(dl_multi.config.settings._SETTINGS, "param_label", dict())    
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_test_tfrecord_file(setting="training"):
    """Write out items that are created by reading tfrecord file."""

    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    dl_multi.tftools.tfrecord.test_tfrecord(
        dl_multi.config.settings.get_data(setting),
        dl_multi.config.settings._SETTINGS["param_specs"],
        dl_multi.config.settings._SETTINGS["param_info"],
        dl_multi.config.settings._SETTINGS["param_io"],
        dl_multi.config.settings._SETTINGS["param_tfrecord"] 
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_test_tfrecords_utils():
    """Test the shortcut functions to convert a standard TensorFlow type to a tf.Example-compatible tf.train.Feature
    """

    import numpy as np

    dl_multi.config.dl_multi.set_cuda_properties(
        glu.get_value(dl_multi.config.settings._SETTINGS, "param_cuda", dict())
    )

    _logger.info("Test the shortcut functions to convert a standard TensorFlow type to a tf.Example-compatible tf.train.Feature")

    # print the results of testing the shortcut functions
    print(dl_multi.tfrecords_utils._bytes_feature(b'test_string'))
    print(dl_multi.tfrecords_utils._bytes_feature(u'test_bytes'.encode('utf-8')))
    print(dl_multi.tftools.tfutils._float_feature(np.exp(1)))
    print(dl_multi.tftools.tfutils._int64_feature(True))
    print(dl_multi.tftools.tfutils._int64_feature(1))
    print(dl_multi.tftools.tfutils._int64_feature(1,True))
