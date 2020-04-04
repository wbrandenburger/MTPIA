# ===========================================================================
#   augmentation.py -----------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import tensorflow as tf

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def rnd_crop_flip(image, annotation, size, minscale, maxscale):

  scale = tf.random_uniform([2], minval=minscale, maxval=maxscale)
  x = tf.to_int32(size[0]/scale[0])
  y = tf.to_int32(size[1]/scale[1])
  crop_size = tf.stack([x, y, 4])  
  
  annotation = tf.to_float(annotation)  
  both = tf.concat([image, annotation], 2)
  
  padx = int(size[0]/2)
  pady = int(size[1]/2)
  both = tf.pad(both, tf.constant([[pady,pady],[padx,padx],[0,0]]))
  
  crop = tf.random_crop(both, crop_size)
  
  img_crop = tf.slice(crop, [0,0,0], [x, y, 3])
  anno_crop = tf.slice(crop, [0,0,3], [x, y, 1])
  
  img_crop = tf.image.resize_images(img_crop, tf.stack(size), method=tf.image.ResizeMethod.BILINEAR)
  anno_crop = tf.image.resize_images(anno_crop, tf.stack(size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])  
  if random_var == 1:
    img_crop = tf.image.flip_left_right(img_crop)
    anno_crop = tf.image.flip_left_right(anno_crop)
    
  #img_crop = tf.slice(img_crop, [0,0,0], [size[0], size[1], 3])
  #anno_crop = tf.slice(anno_crop, [0,0,0], [size[0], size[1], 1])
  
  img_crop = tf.image.resize_image_with_crop_or_pad(img_crop, size[0], size[1])
  anno_crop = tf.image.resize_image_with_crop_or_pad(anno_crop, size[0], size[1])

  return img_crop, anno_crop

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def rnd_crop_flip_height(image, annotation, size, minscale, maxscale):

  scale = tf.random_uniform([2], minval=minscale, maxval=maxscale)
  x = tf.to_int32(size[0]/scale[0])
  y = tf.to_int32(size[1]/scale[1])
  crop_size = tf.stack([x, y, 5])  
  
  annotation = tf.to_float(annotation)  
  both = tf.concat([image, annotation], 2)
  
  padx = int(size[0]/2)
  pady = int(size[1]/2)
  both = tf.pad(both, tf.constant([[pady,pady],[padx,padx],[0,0]]))
  
  crop = tf.random_crop(both, crop_size)
  
  img_crop = tf.slice(crop, [0,0,0], [x, y, 4])
  anno_crop = tf.slice(crop, [0,0,4], [x, y, 1])
  
  img_crop = tf.image.resize_images(img_crop, tf.stack(size), method=tf.image.ResizeMethod.BILINEAR)
  anno_crop = tf.image.resize_images(anno_crop, tf.stack(size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])  
  if random_var == 1:
    img_crop = tf.image.flip_left_right(img_crop)
    anno_crop = tf.image.flip_left_right(anno_crop)
    
  #img_crop = tf.slice(img_crop, [0,0,0], [size[0], size[1], 3])
  #anno_crop = tf.slice(anno_crop, [0,0,0], [size[0], size[1], 1])
  
  img_crop = tf.image.resize_image_with_crop_or_pad(img_crop, size[0], size[1])
  anno_crop = tf.image.resize_image_with_crop_or_pad(anno_crop, size[0], size[1])

  return img_crop, anno_crop

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def rnd_crop_rotate_with_flips_height(image, annotation, size, minscale, maxscale):

  scale = tf.random_uniform([2], minval=minscale, maxval=maxscale)
  x = tf.to_int32(size[0]/scale[0])
  y = tf.to_int32(size[1]/scale[1])
  crop_size = tf.stack([x, y, 5])  
  
  annotation = tf.to_float(annotation)  
  
  padx = int(size[0]/2)
  pady = int(size[1]/2)
  image = tf.pad(image, tf.constant([[pady,pady],[padx,padx],[0,0]]))
  annotation = tf.pad(annotation, tf.constant([[pady,pady],[padx,padx],[0,0]]))
  
  angle = tf.random_uniform([1], minval=-180, maxval=180)
  image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')
  annotation = tf.contrib.image.rotate(annotation, angle, interpolation='NEAREST')  
  
  both = tf.concat([image, annotation], 2)  
  crop = tf.random_crop(both, crop_size)
  
  img_crop = tf.slice(crop, [0,0,0], [x, y, 4])
  anno_crop = tf.slice(crop, [0,0,4], [x, y, 1])
  
  img_crop = tf.image.resize_images(img_crop, tf.stack(size), method=tf.image.ResizeMethod.BILINEAR)
  anno_crop = tf.image.resize_images(anno_crop, tf.stack(size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])  
  if random_var == 1:
    img_crop = tf.image.flip_left_right(img_crop)
    anno_crop = tf.image.flip_left_right(anno_crop)
    
  random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])  
  if random_var == 1:
    img_crop = tf.image.flip_up_down(img_crop)
    anno_crop = tf.image.flip_up_down(anno_crop)
    
  #img_crop = tf.slice(img_crop, [0,0,0], [size[0], size[1], 3])
  #anno_crop = tf.slice(anno_crop, [0,0,0], [size[0], size[1], 1])
  
  img_crop = tf.image.resize_image_with_crop_or_pad(img_crop, size[0], size[1])
  anno_crop = tf.image.resize_image_with_crop_or_pad(anno_crop, size[0], size[1])

  return img_crop, anno_crop
  
#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def rnd_crop_rotate_90_with_flips_height(image, height, annotation, size, minscale, maxscale):

  scale = tf.random_uniform([2], minval=minscale, maxval=maxscale)
  x = tf.to_int32(size[0]/scale[0])
  y = tf.to_int32(size[1]/scale[1])
  crop_size = tf.stack([x, y, 5])  
  
  image = tf.to_float(image)  
  annotation = tf.to_float(annotation)  
  height= tf.to_float(height)
   
  padx = int(size[0]/2)
  pady = int(size[1]/2)
  image = tf.pad(image, tf.constant([[pady,pady],[padx,padx],[0,0]]))
  height = tf.pad(height, tf.constant([[pady,pady],[padx,padx],[0,0]]))
  annotation = tf.pad(annotation, tf.constant([[pady,pady],[padx,padx],[0,0]]))
  
  random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
  if random_var == 1:
    image = tf.image.rot90(image)
    height = tf.image.rot90(height)
    annotation = tf.image.rot90(annotation)
  
  #angle = tf.random_uniform([1], minval=-180, maxval=180)
  #image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')
  #annotation = tf.contrib.image.rotate(annotation, angle, interpolation='NEAREST')  
  
  both = tf.concat([image, height, annotation], 2)  
  crop = tf.random_crop(both, crop_size)
  
  img_crop = tf.slice(crop, [0,0,0], [x, y, 3])
  height_crop = tf.slice(crop, [0,0,3], [x, y, 1])
  anno_crop = tf.slice(crop, [0,0,4], [x, y, 1])
  
  img_crop = tf.image.resize_images(img_crop, tf.stack(size), method=tf.image.ResizeMethod.BILINEAR)
  height_crop = tf.image.resize_images(height_crop, tf.stack(size), method=tf.image.ResizeMethod.BILINEAR)
  anno_crop = tf.image.resize_images(anno_crop, tf.stack(size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  random_var = tf.random_uniform(maxval=4, dtype=tf.int32, shape=[])  
  if random_var == 1:
    img_crop = tf.image.flip_left_right(img_crop)
    height_crop = tf.image.flip_left_right(height_crop)
    anno_crop = tf.image.flip_left_right(anno_crop)    
  #random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])  
  if random_var == 2:
    img_crop = tf.image.flip_up_down(img_crop)
    height_crop = tf.image.flip_up_down(height_crop)
    anno_crop = tf.image.flip_up_down(anno_crop)
  if random_var == 3:
    img_crop = tf.image.flip_left_right(img_crop)
    anno_crop = tf.image.flip_left_right(anno_crop)
    height_crop = tf.image.flip_left_right(height_crop)
    img_crop = tf.image.flip_up_down(img_crop)
    anno_crop = tf.image.flip_up_down(anno_crop)
    height_crop = tf.image.flip_left_right(height_crop)
  #img_crop = tf.slice(img_crop, [0,0,0], [size[0], size[1], 3])
  #anno_crop = tf.slice(anno_crop, [0,0,0], [size[0], size[1], 1])
  
  img_crop = tf.image.resize_image_with_crop_or_pad(img_crop, size[0], size[1]) 
  height_crop = tf.image.resize_image_with_crop_or_pad(height_crop, size[0], size[1])
  anno_crop = tf.image.resize_image_with_crop_or_pad(anno_crop, size[0], size[1])
  return img_crop, height_crop, anno_crop
  
#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def rnd_crop_rotate_with_flips_IR(image, annotation, size, minscale, maxscale):

  scale = tf.random_uniform([2], minval=minscale, maxval=maxscale)
  x = tf.to_int32(size[0]/scale[0])
  y = tf.to_int32(size[1]/scale[1])
  crop_size = tf.stack([x, y, 2])  
  
  annotation = tf.to_float(annotation)  
  
  padx = int(size[0]/2)
  pady = int(size[1]/2)
  image = tf.pad(image, tf.constant([[pady,pady],[padx,padx],[0,0]]))
  annotation = tf.pad(annotation, tf.constant([[pady,pady],[padx,padx],[0,0]]))
  
  angle = tf.random_uniform([1], minval=-180, maxval=180)
  image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')
  annotation = tf.contrib.image.rotate(annotation, angle, interpolation='NEAREST')  
  
  both = tf.concat([image, annotation], 2)  
  crop = tf.random_crop(both, crop_size)
  
  img_crop = tf.slice(crop, [0,0,0], [x, y, 1])
  anno_crop = tf.slice(crop, [0,0,1], [x, y, 1])
  
  img_crop = tf.image.resize_images(img_crop, tf.stack(size), method=tf.image.ResizeMethod.BILINEAR)
  anno_crop = tf.image.resize_images(anno_crop, tf.stack(size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])  
  if random_var == 1:
    img_crop = tf.image.flip_left_right(img_crop)
    anno_crop = tf.image.flip_left_right(anno_crop)
    
  random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])  
  if random_var == 1:
    img_crop = tf.image.flip_up_down(img_crop)
    anno_crop = tf.image.flip_up_down(anno_crop)
    
  #img_crop = tf.slice(img_crop, [0,0,0], [size[0], size[1], 3])
  #anno_crop = tf.slice(anno_crop, [0,0,0], [size[0], size[1], 1])
  
  img_crop = tf.image.resize_image_with_crop_or_pad(img_crop, size[0], size[1])
  anno_crop = tf.image.resize_image_with_crop_or_pad(anno_crop, size[0], size[1])

  return img_crop, anno_crop
    
#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def full_img(image, annotation, size, minscale, maxscale):

  annotation = tf.to_float(annotation) 
  
  random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])  
  if random_var == 1:
    image = tf.image.flip_left_right(image)
    annotation = tf.image.flip_left_right(annotation)
    
    
  both = tf.concat([image, annotation], 2)
  both = tf.image.resize_image_with_crop_or_pad(both, size[0], size[1])

  scale = tf.random_uniform([2], minval=minscale, maxval=maxscale)
  y = tf.to_int32(size[0]*scale[0])
  x = tf.to_int32(size[1]*scale[1])
  
  img_crop = tf.slice(both, [0,0,0], [size[0], size[1], 3])
  anno_crop = tf.slice(both, [0,0,3], [size[0], size[1], 1])
  
  img_res = tf.image.resize_images(img_crop, tf.stack([y, x]), method=tf.image.ResizeMethod.BILINEAR)
  anno_res = tf.image.resize_images(anno_crop, tf.stack([y, x]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  img_crop = tf.image.resize_image_with_crop_or_pad(img_res, size[0], size[1])
  anno_crop = tf.image.resize_image_with_crop_or_pad(anno_res, size[0], size[1])
  
  
  return img_crop, anno_crop

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def add_color(image, alpha=0.25, gamma=40, cwise=0.15):

  cwise = tf.random_uniform([3], 1-cwise, 1+cwise)
  gamma = tf.random_uniform([1],-gamma, gamma)
  alpha = tf.random_uniform([1],1-alpha,1+alpha)
  
  image = tf.multiply(image, alpha)
  image = tf.add(image, gamma)
  image = tf.multiply(image, cwise)
  
  image = tf.clip_by_value(image, 0, 255)  
  
  return image

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------  
def add_noise(image, size, sigma=0.02):

  noise = tf.random_normal([size[0], size[1], 3], 0, sigma)*255
  image = tf.add(image, noise)
  
  image = tf.clip_by_value(image, 0, 255)  
  
  return image