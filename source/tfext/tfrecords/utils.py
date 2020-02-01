# ===========================================================================
#   utils.py ----------------------------------------------------------------
# ===========================================================================

""" The following functions can be used to convert a value to a type compatible with tf.Example.

    The tf.train.Feature message type can accept one of the following three types. Most other generic types can be coerced into one of these:

    tf.train.BytesList : string / byte
    tf.train.FloatList : float (float32) / double (float64)

    tf.train.Int64List : bool / enum / int32 / uint32 / int64 / uint64

    In order to convert a standard TensorFlow type to a tf.Example-compatible tf.train.Feature, you can use the shortcut functions below. Note that each function takes a scalar input value and returns a tf.train.Feature containing one of the three list types above.
"""

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import tensorflow as tf

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def _bytes_feature(value, serialize=False):
    """Returns a bytes_list from a string / byte.

    Parameters
    ----------
    value : string / byte

    Returns
    -------
    feature : bytes_list
        Converted value compatible with tf.Example.
    """
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return feature if not serialize else feature.SerializeToString()

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def _float_feature(value, serialize=False):
    """Returns a float_list from a float / double.

    Parameters
    ----------
    value : float / double

    Returns
    -------
    feature : float_list
        Converted value compatible with tf.Example.
    """
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    return feature if not serialize else feature.SerializeToString()

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def _int64_feature(value, serialize=False):
    """Returns an int64_list from a bool / enum / int / uint.

    Parameters
    ----------
    value : double bool / enum / int / uint

    Returns
    -------
    feature : int64_list
        Converted value compatible with tf.Example.
    """  
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    return feature if not serialize else feature.SerializeToString()