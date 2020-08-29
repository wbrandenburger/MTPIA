import tensorflow as tf  

def layer(x, filters_per_step=12, droprate=0.2, name=""):
  
  with tf.name_scope(name):
    bn = tf.layers.batch_normalization(x, training=True, name=name+'_bn')
    relu = tf.nn.relu(bn, name=name+'_relu')
    
    conv = tf.layers.conv2d(
      inputs=relu,
      filters=filters_per_step,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      name=name+'_conv')

    if droprate>0:
      conv = tf.layers.dropout(conv, droprate, name=name+'_drop')

  return conv
  
# end layer

def dense_block(x, steps=4, filters_per_step=12, droprate=0.2, name=""):

  dense_out = []
  with tf.name_scope(name):
    for i in range(steps):
      conv = layer(x, filters_per_step, droprate, name=name+'_layer_'+str(i))
      x = tf.concat([x, conv], axis=3)
      dense_out.append(conv)

  return tf.concat(dense_out, axis=3)
  
# end dense_block


def transition_down(inp, filters, droprate=0.2, name=""):

  with tf.name_scope(name):
    bn = tf.layers.batch_normalization(inp, training=True, name=name+'_bn')
    relu = tf.nn.relu(bn, name=name+'_relu')
    
    conv = tf.layers.conv2d(
      inputs=relu,
      filters=filters,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      name=name+'_conv')
      
    if droprate>0:
      conv = tf.layers.dropout(conv, droprate, name=name+'_drop')
      
    pool = tf.layers.max_pooling2d(
      inputs=conv,
      pool_size=[2,2],
      strides=[2,2],
      name=name+'_pool')

  return pool
  
# end transition_down


def transition_up(inp, filters, name=""):
  
  with tf.name_scope(name):
    up = tf.layers.conv2d_transpose(
      inputs=inp,
      filters=filters,
      kernel_size=[3, 3],
      strides=[2,2],
      padding='same',
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      name=name+'_deconv')

  return up
  
# end transition_up

def multi_task_classification_regression(vaihingen_batch):
  
  concats = []
  
  with tf.compat.v1.variable_scope('encoder'):  
   
    x = tf.layers.conv2d(
      inputs=vaihingen_batch,
      filters=48,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      name='vaihingen_initial_conv')
    
    # downscale
    for block in range(5):
      dense = dense_block(x, 4, 12, 0.2, name='down_db_'+str(block))
      x = tf.concat([x, dense], axis=3, name='down_concat_'+str(block))
      concats.append(x)
      x = transition_down(x, x.get_shape()[-1], 0.2, name='td_'+str(block))
    
    # bottleneck block
    x = dense_block(x, 4, 12, 0.2, name='bottleneck')
  
  with tf.compat.v1.variable_scope('decoder'):
    # upscale
    for i, block_nb in enumerate(range(5, 1, -1)):
      x = transition_up(x, x.get_shape()[-1], name='tu_'+str(block_nb))
      x = tf.concat([x, concats[len(concats) - i - 1]], axis=3, name='up_concat_'+str(block_nb))
      x = dense_block(x, 4, 12, 0.2, name='up_db_'+str(block_nb))
      
    # separate dense block for etrims and vaihingen
    x = transition_up(x, x.get_shape()[-1], name='tu_0')
    x = tf.concat([x, concats[len(concats) - 4 - 1]], axis=3, name='up_concat_0')
    
    x_vaihingen = dense_block(x, 4, 12, 0.2, name='up_db_0_vaihingen')
    dsm_vaihingen = dense_block(x, 4, 12, 0.2, name='up_db_0_vaihingen_dsm')
      
  with tf.compat.v1.variable_scope('prediction_vaihingen'):
    
    x_vaihingen_pred1 = tf.layers.conv2d(
      inputs=x_vaihingen,
      filters=6,#32,
      kernel_size=[1, 1],
      padding="same",
      #activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      name='pred1_conv_vaihingen')
      
    #x_vaihingen_pred1 = tf.layers.dropout(x_vaihingen_pred1, 0.2, name='pred1_conv_vaihingen_drop')
  
    #x_vaihingen_pred2 = tf.layers.conv2d(
    #  inputs=x_vaihingen_pred1,
    #  filters=6,
    #  kernel_size=[1, 1],
    #  padding="same",
    #  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #  name='pred2_conv_vaihingen')
      
  with tf.compat.v1.variable_scope('regression_vaihingen'):

    x_vaihingen_reg1 = tf.layers.conv2d(
      inputs=dsm_vaihingen,
      filters=1,#32,
      kernel_size=[1, 1],
      padding="same",
      # activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      name='reg1_conv_vaihingen')
      
    #x_vaihingen_reg1 = tf.layers.dropout(x_vaihingen_reg1, 0.2, name='reg1_conv_vaihingen_drop')
  
    #x_vaihingen_reg2 = tf.layers.conv2d(
    #  inputs=x_vaihingen_reg1,
    #  filters=1,
    #  kernel_size=[1, 1],
    #  padding="same",
    #  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #  name='reg2_conv_vaihingen')

  # argmax_vaihingen = tf.argmax(x_vaihingen_pred1, axis=-1, name='argmax_vaihingen')

  return x_vaihingen_pred1, x_vaihingen_reg1 #, argmax_vaihingen, 


def single_task_regression(vaihingen_batch):
  
  concats = []
  
  with tf.compat.v1.variable_scope('encoder'):  
   
    x = tf.layers.conv2d(
      inputs=vaihingen_batch,
      filters=48,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      name='vaihingen_initial_conv')
    
    # downscale
    for block in range(5):
      dense = dense_block(x, 4, 12, 0.2, name='down_db_'+str(block))
      x = tf.concat([x, dense], axis=3, name='down_concat_'+str(block))
      concats.append(x)
      x = transition_down(x, x.get_shape()[-1], 0.2, name='td_'+str(block))
    
    # bottleneck block
    x = dense_block(x, 4, 12, 0.2, name='bottleneck')
  
  with tf.compat.v1.variable_scope('decoder'):
    # upscale
    for i, block_nb in enumerate(range(5, 1, -1)):
      x = transition_up(x, x.get_shape()[-1], name='tu_'+str(block_nb))
      x = tf.concat([x, concats[len(concats) - i - 1]], axis=3, name='up_concat_'+str(block_nb))
      x = dense_block(x, 4, 12, 0.2, name='up_db_'+str(block_nb))
      
    # separate dense block for etrims and vaihingen
    x = transition_up(x, x.get_shape()[-1], name='tu_0')
    x = tf.concat([x, concats[len(concats) - 4 - 1]], axis=3, name='up_concat_0')
    
    #x_vaihingen = dense_block(x, 4, 12, 0.2, name='up_db_0_vaihingen')
    dsm_vaihingen = dense_block(x, 4, 12, 0.2, name='up_db_0_vaihingen_dsm')
      
  # with tf.compat.v1.variable_scope('prediction_vaihingen'):
    
  #   x_vaihingen_pred1 = tf.layers.conv2d(
  #     inputs=x_vaihingen,
  #     filters=6,#32,
  #     kernel_size=[1, 1],
  #     padding="same",
  #     #activation=tf.nn.relu,
  #     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
  #     name='pred1_conv_vaihingen')
      
    #x_vaihingen_pred1 = tf.layers.dropout(x_vaihingen_pred1, 0.2, name='pred1_conv_vaihingen_drop')
  
    #x_vaihingen_pred2 = tf.layers.conv2d(
    #  inputs=x_vaihingen_pred1,
    #  filters=6,
    #  kernel_size=[1, 1],
    #  padding="same",
    #  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #  name='pred2_conv_vaihingen')
      
  with tf.compat.v1.variable_scope('regression_vaihingen'):
    
    x_vaihingen_reg1 = tf.layers.conv2d(
      inputs=dsm_vaihingen,
      filters=1,#32,
      kernel_size=[1, 1],
      padding="same",
      # activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      name='reg1_conv_vaihingen')
      
    #x_vaihingen_reg1 = tf.layers.dropout(x_vaihingen_reg1, 0.2, name='reg1_conv_vaihingen_drop')
  
    #x_vaihingen_reg2 = tf.layers.conv2d(
    #  inputs=x_vaihingen_reg1,
    #  filters=1,
    #  kernel_size=[1, 1],
    #  padding="same",
    #  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #  name='reg2_conv_vaihingen')

  # argmax_vaihingen = tf.argmax(x_vaihingen_pred1, axis=-1, name='argmax_vaihingen')

  return x_vaihingen_reg1

def single_task_classification(vaihingen_batch):
  
  concats = []
  
  with tf.compat.v1.variable_scope('encoder'):  
   
    x = tf.layers.conv2d(
      inputs=vaihingen_batch,
      filters=48,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      name='vaihingen_initial_conv')
    
    # downscale
    for block in range(5):
      dense = dense_block(x, 4, 12, 0.2, name='down_db_'+str(block))
      x = tf.concat([x, dense], axis=3, name='down_concat_'+str(block))
      concats.append(x)
      x = transition_down(x, x.get_shape()[-1], 0.2, name='td_'+str(block))
    
    # bottleneck block
    x = dense_block(x, 4, 12, 0.2, name='bottleneck')
  
  with tf.compat.v1.variable_scope('decoder'):
    # upscale
    for i, block_nb in enumerate(range(5, 1, -1)):
      x = transition_up(x, x.get_shape()[-1], name='tu_'+str(block_nb))
      x = tf.concat([x, concats[len(concats) - i - 1]], axis=3, name='up_concat_'+str(block_nb))
      x = dense_block(x, 4, 12, 0.2, name='up_db_'+str(block_nb))
      
    # separate dense block for etrims and vaihingen
    x = transition_up(x, x.get_shape()[-1], name='tu_0')
    x = tf.concat([x, concats[len(concats) - 4 - 1]], axis=3, name='up_concat_0')
    
    x_vaihingen = dense_block(x, 4, 12, 0.2, name='up_db_0_vaihingen')
    # dsm_vaihingen = dense_block(x, 4, 12, 0.2, name='up_db_0_vaihingen_dsm')
      
  with tf.compat.v1.variable_scope('prediction_vaihingen'):
    
    x_vaihingen_pred1 = tf.layers.conv2d(
      inputs=x_vaihingen,
      filters=6,#32,
      kernel_size=[1, 1],
      padding="same",
      #activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      name='pred1_conv_vaihingen')
      
    #x_vaihingen_pred1 = tf.layers.dropout(x_vaihingen_pred1, 0.2, name='pred1_conv_vaihingen_drop')
  
    #x_vaihingen_pred2 = tf.layers.conv2d(
    #  inputs=x_vaihingen_pred1,
    #  filters=6,
    #  kernel_size=[1, 1],
    #  padding="same",
    #  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #  name='pred2_conv_vaihingen')
      
  # # with tf.compat.v1.variable_scope('regression_vaihingen'):
    
  # #   x_vaihingen_reg1 = tf.layers.conv2d(
  # #     inputs=dsm_vaihingen,
  # #     filters=1,#32,
  # #     kernel_size=[1, 1],
  # #     padding="same",
  # #     #activation=tf.nn.relu,
  # #     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
  # #     name='reg1_conv_vaihingen')
      
    #x_vaihingen_reg1 = tf.layers.dropout(x_vaihingen_reg1, 0.2, name='reg1_conv_vaihingen_drop')
  
    #x_vaihingen_reg2 = tf.layers.conv2d(
    #  inputs=x_vaihingen_reg1,
    #  filters=1,
    #  kernel_size=[1, 1],
    #  padding="same",
    #  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #  name='reg2_conv_vaihingen')

  argmax_vaihingen = tf.argmax(x_vaihingen_pred1, axis=-1, name='argmax_vaihingen')

  return x_vaihingen_pred1 , argmax_vaihingen


