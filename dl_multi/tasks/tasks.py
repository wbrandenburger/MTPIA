# ===========================================================================
#   tasks.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.config.settings
import dl_multi.utils.format

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_default():
    task_print_user_settings()

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

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_test_tfrecords_utils():
    """Test the shortcut functions to convert a standard TensorFlow type to a tf.Example-compatible tf.train.Feature
    """


    dl_multi.__init__._logger.info("Test the shortcut functions to convert a standard TensorFlow type to a tf.Example-compatible tf.train.Feature")

    import dl_multi.tfext.tfrecords.utils
    import numpy as np


    # print the results of testing the shortcut functions
    print(dl_multi.tfext.tfrecords.utils._bytes_feature(b'test_string'))
    print(dl_multi.tfext.tfrecords.utils._bytes_feature(u'test_bytes'.encode('utf-8')))

    print(dl_multi.tfext.tfrecords.utils._float_feature(np.exp(1)))

    print(dl_multi.tfext.tfrecords.utils._int64_feature(True))
    print(dl_multi.tfext.tfrecords.utils._int64_feature(1))

    print(dl_multi.tfext.tfrecords.utils._int64_feature(1,True))