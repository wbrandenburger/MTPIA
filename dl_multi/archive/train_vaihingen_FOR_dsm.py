import tensorflow as tf
import os


# bei einzelner GPU '0', sonst z.B. [1,3]
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tf_records_vaihingen import read_tfrecord_and_decode_into_image_annotation_pair_tensors_vaihingen_dsm
from tiramisu56_vaihingen_FOR_dsm import tiramisu56
from augmentation import rnd_crop_rotate_90_with_flips_dsm

slim = tf.contrib.slim

batch_size_vaihingen=2
image_train_size = [448, 448]
num_epochs = 150000 ## eigentlich num_iterations, aber tensorflow braucht(e) num_epochs

tfrecords_filename_vaihingen = '/media/Raid/matthias/tensorflow/PIA2019/vaihingen_w_dsm.tfrecords'

name = 'PIA/vaihingen_FOR_dsm_190711_DLBox_wd0_09'
checkpoints_dir = '/media/Raid/matthias/tensorflow/PIA2019/checkpoints/'+name
log_folder = '/media/Raid/matthias/tensorflow/PIA2019/log/'+name

##########################################################

filename_queue_vaihingen = tf.train.string_input_producer([tfrecords_filename_vaihingen])

image_vaihingen, annotation_vaihingen = read_tfrecord_and_decode_into_image_annotation_pair_tensors_vaihingen_dsm(filename_queue_vaihingen)

image_vaihingen = tf.to_float(image_vaihingen)
annotation_vaihingen = annotation_vaihingen + 1


data_vaihingen, annotation_vaihingen = rnd_crop_rotate_90_with_flips_dsm(image_vaihingen, annotation_vaihingen, image_train_size, 0.95, 1.1)
image_vaihingen = data_vaihingen[:,:,0:3] / 127.5 - 1.
dsm_vaihingen =  tf.image.per_image_standardization(data_vaihingen[:,:,3:4])


## concat für single-task mit 4-channel input
#image_vaihingen = tf.concat([image_vaihingen, tf.expand_dims(dsm_vaihingen, axis=2)], axis=2)


##### vaihingen
# Create batches of 'batch size'  images, labels and dsm by randomly shuffling tensors. The capacity specifies the maximum of elements in the queue
# @todo[generalize]:
image_batch_vaihingen, annotation_batch_vaihingen, dsm_batch_vaihingen = tf.train.shuffle_batch(
    [image_vaihingen, annotation_vaihingen, dsm_vaihingen],
    batch_size=batch_size_vaihingen,
    capacity=64,
    min_after_dequeue=32,
    num_threads=16
)
                                             
with tf.compat.v1.variable_scope("net"):
  pred_vaihingen, argmax_vaihingen, reg_vaihingen = tiramisu56(image_batch_vaihingen)

mask_vaihingen = tf.to_float(tf.squeeze(tf.greater(annotation_batch_vaihingen, 0.)))
labels_vaihingen = tf.to_int32(tf.squeeze(tf.maximum(annotation_batch_vaihingen-1, 0), axis=3))

pred_loss_vaihingen = tf.reduce_mean(tf.compat.v1.losses.compute_weighted_loss(
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_vaihingen, logits=pred_vaihingen),
    weights = tf.to_float(mask_vaihingen)))

acc_vaihingen = 1 - ( tf.count_nonzero((tf.to_float(argmax_vaihingen)-tf.to_float(labels_vaihingen)), dtype=tf.float32)
              / (image_train_size[0] * image_train_size[1] * batch_size_vaihingen ))
              
reg_loss_vaihingen = tf.compat.v1.losses.mean_squared_error(dsm_batch_vaihingen, reg_vaihingen, weights = tf.expand_dims(mask_vaihingen, axis=3))


task_weight = 0.9
loss_vaihingen = task_weight * pred_loss_vaihingen + (1. - task_weight) * reg_loss_vaihingen


### end if vaihingen


update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_step_both = tf.contrib.opt.AdamWOptimizer(0).minimize(loss_vaihingen)
      
tf.summary.scalar('loss_vaihingen', loss_vaihingen)
tf.summary.scalar('accuracy_vaihingen', acc_vaihingen)

tf.summary.scalar('pred_loss_vaihingen', pred_loss_vaihingen)
tf.summary.scalar('reg_loss_vaihingen', reg_loss_vaihingen)


merged_summary_op = tf.summary.merge_all()
summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist log_folderyet
if not os.path.exists(log_folder):
     os.makedirs(log_folder)

# The op for initializing the variables.
init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                  tf.compat.v1.local_variables_initializer())
                   
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:

  sess.run(init_op)
    
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
    

  loss_v = 0
  acc_v = 0
  loss_v_r = 0


  # iterate epochs (eigentlich iterations!!!)
  for i in range(num_epochs+1):

    loss_v, loss_v_r, acc_v, summary_string, _ = sess.run([ pred_loss_vaihingen, reg_loss_vaihingen, acc_vaihingen, merged_summary_op, train_step_both])
    
    summary_string_writer.add_summary(summary_string, i)
        
    print("Step: " + str(i) + " Loss_PRED: " + str(loss_v) + " Loss_REG: " + str(loss_v_r) + " Accuracy_V: " + str(acc_v) )
        
    if i % 100 == 0:
      save_path = saver.save(sess, checkpoints_dir+"/pia.ckpt")
      print("Model saved in file: %s" % save_path)
    if i % 25000 == 0:
      save_path = saver.save(sess, checkpoints_dir+"/pia.ckpt", global_step=i)
      print("Model saved in file: %s" % save_path)


  coord.request_stop()
  coord.join(threads)
    
  save_path = saver.save(sess, checkpoints_dir+"/pia.ckpt", global_step=i)
  print("Model saved in file: %s" % save_path)
    
summary_string_writer.close()
