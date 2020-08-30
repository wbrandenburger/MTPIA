# ===========================================================================
#   augmentation.py -----------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import tensorflow as tf

# TODO: Problems with tfaugmentation only when calling multi-task environmens

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def rnd_crop(
        tensors, 
        size,
        channels,  
        scale, 
        rescale=[1.0, 1.0], 
        flip=False,
        rotate=False
    ):

    channel = 0
    for idx in range(len(tensors)):
        tensors[idx] = tf.to_float(tensors[idx])
        channel += channels[idx]

    rand_rescale = tf.random.uniform([2], minval=rescale[0], maxval=rescale[1])
    x = tf.to_int32(size[0]/rand_rescale[0])
    y = tf.to_int32(size[1]/rand_rescale[1])
    channel = tf.to_int32(channel)

    size_crop = tf.stack([x, y, channel])

    x_pad = int(size[0]/2)
    y_pad = int(size[1]/2)
    crop = tf.image.random_crop(tf.pad(tf.concat(tensors, 2), tf.constant([[y_pad, y_pad], [x_pad, x_pad], [0,0]])), size_crop)
    
    for idx in range(len(tensors)):
        tensors[idx] = tf.slice(
            crop, [0, 0, 0 if idx == 0 else channels[idx-1]], 
            [x, y, channels[idx]]
        )

        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if scale[idx] == "nominal" else tf.image.ResizeMethod.BILINEAR
        tensors[idx] = tf.image.resize(tensors[idx], tf.stack(size), method=method)

        # flip tensors
        if flip: tensors[idx] = flip_tensor(tensors[idx]) 
        if rotate: tensors[idx] = rotate_tensor(tensors[idx], scale[idx])

        tensors[idx] = tf.image.resize_with_crop_or_pad(tensors[idx], size[0], size[1])

    return tensors

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def flip_tensor(tensor):
    rand_val = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[]) 
    if rand_val == 1:
        tensor = tf.image.flip_left_right(tensor)

    rand_val = tf.random.uniform(maxval=2, dtype=tf.int32, shape=[]) 
    if rand_val == 1:
        tensor =tf.image.flip_up_down(tensor)
    
    return tensor

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def rotate_tensor(tensor, scale):
    angle = tf.random.uniform([1], minval=-180, maxval=180)
    method = "NEAREST" if scale == "nominal" else "BILINEAR"

    tensor = tf.contrib.image.rotate(tensor, angle, interpolation=method)
    
    return tensor

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def add_color(tensor, alpha=0.25, gamma=40, cwise=0.15):

  cwise = tf.random.uniform([3], 1-cwise, 1+cwise)
  gamma = tf.random.uniform([1],-gamma, gamma)
  alpha = tf.random.uniform([1],1-alpha,1+alpha)
  
  tensor = tf.multiply(tensor, alpha)
  tensor = tf.add(tensor, gamma)
  tensor = tf.multiply(tensor, cwise)
  
  tensor = tf.clip_by_value(tensor, 0, 255)  
  
  return tensor

  #   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------  
def add_noise(tensor, size, sigma=0.02):

  noise = tf.random_normal([size[0], size[1], 3], 0, sigma)*255
  tensor = tf.add(tensor, noise)
  
  tensor = tf.clip_by_value(tensor, 0, 255)  
  
  return tensor