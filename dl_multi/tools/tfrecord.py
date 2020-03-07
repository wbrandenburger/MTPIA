# ===========================================================================
#   tfrecords_utils.py-------------------------------------------------------
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
import dl_multi.__init__

import numpy as np
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

    
def read_tfrecord(tfrecord_queue):
    """Return image/annotation tensors that are created by reading tfrecord file.

    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped image/annotation tensors.

    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue
        String queue object from tf.train.string_input_producer()
    
    Returns
    -------
    image, annotation : tuple of tf.int32 (image, annotation)
        Tuple of image/annotation tensors
    """
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'data_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    
    image = tf.decode_raw(features['data_raw'], tf.float32)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 4])
    annotation_shape = tf.stack([height, width, 1])
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    
    return image, annotation  

def write_tfrecord(files, specs, output):
    """Create a dictionary with features that may be relevant."""

    dl_multi.__init__._logger.debug("Start creation of tfrecors with settings:\n'output':\t'{}'".format(output))

    img_set, _ = dl_multi.tools.data.get_data(files)
    with tf.io.TFRecordWriter(output["tf_records"]) as writer:
        for item in iter(img_set):
            dl_multi.__init__._logger.debug("Processing image '{}'".format(item[0].path))

            image = np.stack((item.index("image").data, item.index("height").data), axis=2)
            label = item.index("label").data

            tf_example = get_tfrecord_features(image, label)
            writer.write(tf_example.SerializeToString())

def get_tfrecord_features(image_string, mask_string):
    """Create a dictionary with features that may be relevant."""
    # tf.enable_eager_execution()
    image_shape = tf.image.decode_jpeg(image_string).shape    
    
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'data_raw': _bytes_feature(image_string),
        'mask_raw': _bytes_feature(mask_string),
    }

    image_features = tf.train.Example(features=tf.train.Features(feature=feature))
