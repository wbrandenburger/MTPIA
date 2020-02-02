# ===========================================================================
#   default.py --------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import __init__
import config.settings
import utils.format
#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():
    
    print_user_settings()
    
    test_tfrecords_utils()


#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def test_user_settings():
    """Print the user settings"""
    
    # print user's defined settings
    __init__._logger.info("Print user's defined settings")
    utils.format.print_data(config.settings._SETTINGS)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def test_user_data():
    """Print the user data"""
    
    # print user's defined data
    __init__._logger.info("Print user's defined data")
    utils.format.print_data(config.settings._DATA)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def test_tfrecords_utils():
    """Test the shortcut functions to convert a standard TensorFlow type to a tf.Example-compatible tf.train.Feature
    """


    __init__._logger.info("Test the shortcut functions to convert a standard TensorFlow type to a tf.Example-compatible tf.train.Feature")

    import tfext.tfrecords.utils
    import numpy as np


    # print the results of testing the shortcut functions
    print(tfext.tfrecords.utils._bytes_feature(b'test_string'))
    print(tfext.tfrecords.utils._bytes_feature(u'test_bytes'.encode('utf-8')))

    print(tfext.tfrecords.utils._float_feature(np.exp(1)))

    print(tfext.tfrecords.utils._int64_feature(True))
    print(tfext.tfrecords.utils._int64_feature(1))

    print(tfext.tfrecords.utils._int64_feature(1,True))