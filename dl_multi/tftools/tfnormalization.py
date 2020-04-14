# ===========================================================================
#   tfnormalization.py --------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import tensorflow as tf

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def default(x):
    return x

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def add_value(x, value=0):
    return x + value

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def standardize_per_input(x, source_range=list(), dest_range=[-1,1], dtype=tf.float32):
    """ Linearly scales each image in image to have mean 0 and variance 1.
    https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    """
    x_normed = tf.cast(x, dtype)
    
    if not source_range:
        source_range = [tf.math.reduce_min(x_normed), tf.math.reduce_max(x_normed)]

    x_normed = (x_normed-source_range[0]) / (source_range[1]-source_range[0])
    x_normed = x_normed * (dest_range[1]-dest_range[0]) + dest_range[0]

    return x_normed

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def normalize_per_input(x, axes=[0,1,2], epsilon=1e-8, dtype=tf.float32):
    x_normed = tf.cast(x, dtype)
    
    mean, variance = tf.nn.moments(x_normed, axes=axes)
    x_normed = (x_normed - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero

    return x_normed