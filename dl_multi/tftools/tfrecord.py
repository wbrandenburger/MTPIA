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
import dl_multi.utils.imgio
from dl_multi.utils import imgtools

import numpy as np
import pathlib
import tensorflow as tf
import tifffile

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

# Create a dictionary describing the features. The key of the dict should be the same with the key in writing function.

_feature_specs = {
    "features" : {
        "rows": tf.io.FixedLenFeature([], tf.int64),
        "cols": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string)
    }, 
    "images" : [
        {"spec": "image", "channels": 3, "type" : tf.uint8, "ext": ".tif"},
        {"spec": "height", "channels": 1, "type" : tf.float32, "ext": ".tif"},
        {"spec": "label", "channels": 1, "type" : tf.uint8, "ext": ".tif"} 
    ]
}
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

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def write_tfrecord(files, specs, param_label=dict(), param_out = dict()):
    """Create a dictionary with features that may be relevant."""

    dl_multi.__init__._logger.debug("Start creation of tfrecors with settings:\n'param_out':\t'{}'".format(param_out))
    
    img_set, _ = dl_multi.utils.imgio.get_data(files, specs=specs)
    with tf.io.TFRecordWriter(param_out["tfrecords"]) as writer:
        for item in iter(img_set):
            dl_multi.__init__._logger.debug("Processing image '{}'".format(item[0].path))

            img = item.spec("image").data
            tf_example = get_tfrecord_features(
                img.shape,
                img.tostring(),
                item.spec("height").data.tostring(),
                imgtools.labels_to_image(item.spec("label").data, param_label).tostring()
            )
            writer.write(tf_example.SerializeToString())

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_tfrecord_features(shape, image_string, height_string, mask_string):
    """Create a dictionary with features that may be relevant."""

    # image_shape = tf.image.decode_jpeg(image_string).shape    
    
    # Create a dictionary describing the features. The key of the dict should be the same with the key in writing function.
    feature = {
        "rows": _int64_feature(shape[0]),
        "cols": _int64_feature(shape[1]),
        "image": _bytes_feature(image_string),
        "height": _bytes_feature(height_string),
        "label": _bytes_feature(mask_string),
    }

    return tf.train.Example(
        features=tf.train.Features(
            feature=feature)
        )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def read_tfrecord_queue(tfrecord_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_queue)

    return get_img_from_tf_features_list(
        tf.io.parse_single_example(serialized_example, features=_feature_specs["features"]), 
        _feature_specs["images"]
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def print_tfrecord(specs, param_out = dict()):

    # Use dataset API to import date directly from TFRecord file.
    data_raw = tf.data.TFRecordDataset(param_out["tfrecords"])

    # Define the parse function to extract a single example as a dict.
    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, _feature_specs["features"])
    data_parsed = data_raw.map(_parse_image_function)
    
    # If there are more than one example, use a for loop to read them out.
    path = pathlib.Path("B:\\DLMulti\\images")
    path.mkdir(parents=True, exist_ok=True)

    for count, features in  enumerate(data_parsed):
        write_img_from_tf_features_list(
            features, 
            _feature_specs["images"],
            path, count
        )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def write_img_from_tf_features_list(features, features_list, path, count):
    for item in features_list:
        tifffile.imwrite(
            path / "{}_{}{}".format(item["spec"], count, item["ext"]), 
            get_img_from_tf_features(
                features[item["spec"]], item["channels"], item["type"],
                tf.cast(features["rows"], tf.int32), 
                tf.cast(features["cols"], tf.int32)
            ).numpy()
        )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_img_from_tf_features_list(features, features_list):
    return [
        get_img_from_tf_features(
            features[item["spec"]], item["channels"], item["type"],
            tf.cast(features["rows"], tf.int32), 
            tf.cast(features["cols"], tf.int32)
        ) for item in features_list 
    ]

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_img_from_tf_features(features, channels, dtype, rows, cols):
    return tf.reshape(
        tf.decode_raw(features, dtype), 
        tf.stack([rows, cols, channels])
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def read_tfrecord_attempt(tfrecord_queue):
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

    # Create a dictionary describing the features. The key of the dict should be the same with the key in writing function.
    features = tf.io.parse_single_example(
      serialized_example,
      features={
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'data_raw': tf.io.FixedLenFeature([], tf.string),
        'mask_raw': tf.io.FixedLenFeature([], tf.string)
        }
    )

    image = tf.decode_raw(features['data_raw'], tf.float32)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 4])
    annotation_shape = tf.stack([height, width, 1])
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    
    return image, annotation  