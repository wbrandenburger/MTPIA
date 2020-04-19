# ===========================================================================
#   tfrecords_utils.py-------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from  dl_multi.__init__ import _logger
import dl_multi.tftools.tfutils
import dl_multi.utils.general as glu
import dl_multi.utils.imgio
from dl_multi.utils import imgtools
from dl_multi.tftools.tftypes import tftypes

import numpy as np
import pathlib
import tensorflow as tf
import tifffile

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
class tfrecord():

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __init__(self, tfrecord, param_info, param_input, param_output):
        self._tfrecord = tfrecord
        
        self._param_info = param_info
        self._param_input = param_input if isinstance(param_input, list) else [param_input]
        self._param_output = param_output if isinstance(param_output, list) else[param_output]

        self._specs = [item["spec"] for item in self._param_input] + [item["spec"] for item in self._param_output]

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_data(self):
        return read_tfrecord(self._tfrecord, self._param_info, specs=self._specs)

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_spec_list(self):
        return self._specs

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_spec_item_list(self, item):
        return [self._param_info[spec][item] for spec in self._specs]

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_data(tfrecord, param_info, param_input, param_output):
    return read_tfrecord(tfrecord, param_info, specs=get_spec_list(param_info, param_input, param_output))

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_spec_list(param_info, param_input, param_output):
    param_input = param_input if isinstance(param_input, list) else [param_input]
    param_output = param_output if isinstance(param_output, list) else[param_output]
    
    specs = [item["spec"] for item in param_input] + [item["spec"] for item in param_output]

    return specs

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_spec_item_list(item, param_info, param_input, param_output):
    specs = get_spec_list(param_info, param_input, param_output)
    return [param_info[spec][item] for spec in specs]

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def write_tfrecord(files, param_specs, param_tfrecord, param_label=dict()):
    """Create a new tfrecord file."""

    _logger.info("Start creation of tfrecords with settings:\nparam_specs:\t{}\nparam_tfrecord:\t{}\nparam_label:\t{}".format(param_specs, param_tfrecord, param_label))      

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    img_in = dl_multi.utils.imgio.get_data(files, param_specs, param_label=param_label)
    
    tfrecord_file = glu.Folder().set_folder(**param_tfrecord["tfrecord"])
  
    #   execution -----------------------------------------------------------
    # -----------------------------------------------------------------------  
    _logger.info("[SAVE] '{}'".format(tfrecord_file))
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for data_set in img_in:
            
            # Create a dictionary describing the features. The key of the dict should be the same with the key in writing function.
            shape = data_set.spec("image").data.shape
            features = {
                "rows": dl_multi.tftools.tfutils._int64_feature(shape[0]),
                "cols": dl_multi.tftools.tfutils._int64_feature(shape[1])
            }
            
            for data_item in data_set:
                img = data_item.data
                features["c-{}".format(data_item.spec)] = dl_multi.tftools.tfutils._int64_feature(img.shape[2] if len(img.shape)>2 else 1)
                features[data_item.spec] = dl_multi.tftools.tfutils._bytes_feature(img.tostring()) 

            writer.write(tf.train.Example(
                features=tf.train.Features(feature=features)
            ).SerializeToString())

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def read_tfrecord(tfrecord, param_info, specs=dict()):
    """Read from a tfrecord file."""

    _logger.info("Start reading from a tfrecord file with settings:\nparam_tfrecord:\t{}\nparam_label:\t{}".format(param_info, specs))

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    tfrecord_queue = tf.train.string_input_producer(
        [glu.Folder().set_folder(**tfrecord)])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_queue)
    
    spec_list = list(param_info.keys())
    spec_sub_list = specs if specs else spec_list

    #   execution -----------------------------------------------------------
    # ----------------------------------------------------------------------- 
    data_raw = tf.io.parse_single_example(serialized_example, features=get_features(spec_list))

    rows = tf.cast(data_raw["rows"], tf.int32)
    cols = tf.cast(data_raw["cols"], tf.int32)
    
    tensors = [ tf.reshape(
            tf.decode_raw(data_raw[spec], tftypes[param_info[spec]["dtype"]]), 
            tf.stack([rows, cols, tf.cast(data_raw["c-{}".format(spec)], tf.int32)])
        ) for spec in spec_sub_list]

    return tensors

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def test_tfrecord(files, param_specs, param_info, param_io, param_tfrecord):
    """Write out items that are created by reading tfrecord file."""

    tf.compat.v1.enable_eager_execution()

    _logger.info("Write out items of a tfrecord file with settings:\nparam_specs:\t{}\nparam_info:\t{}\nparam_io:\t{}\nparam_tfrecord:\t{}".format(param_specs, param_info, param_io, param_tfrecord))          

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    img_in, img_out,_ , _ = dl_multi.utils.imgio.get_data(files, param_specs, param_io=param_io)
  
    tfrecord_file = glu.Folder().set_folder(**param_tfrecord["tfrecord"])
  
    #   execution -----------------------------------------------------------
    # -----------------------------------------------------------------------

    # Use dataset API to import date directly from TFRecord file.
    data = tf.data.TFRecordDataset(tfrecord_file)

    # Define the parse function to extract a single example as a dict.
    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, get_features(param_specs))

    data_parsed = data.map(_parse_image_function)
    
    for data_set, data_raw in zip(img_in, data_parsed):
        rows = tf.cast(data_raw["rows"], tf.int32)
        cols = tf.cast(data_raw["cols"], tf.int32)
        for data_item, data_spec in zip(data_set, param_specs):
            channels = tf.cast(data_raw["c-{}".format(data_spec)], tf.int32)
            _logger.info("[SHAPE] {}, {}, {}".format(rows.numpy(), cols.numpy(), channels.numpy()))

            dtype = tftypes[param_info[data_spec]["dtype"]]
            img = tf.reshape(tf.decode_raw(data_raw[data_spec], dtype), tf.stack([rows, cols, channels]))
            img_out(data_item.path, img.numpy(), prefix=data_spec)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_features(param_specs):
    features = {
        "rows": tf.io.FixedLenFeature([], tf.int64),
        "cols": tf.io.FixedLenFeature([], tf.int64),
    }
    for spec in param_specs:
        features["c-{}".format(spec)] = tf.io.FixedLenFeature([], tf.int64)
        features[spec] = tf.io.FixedLenFeature([], tf.string)
    
    return features