# ===========================================================================
#   train.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger 
import dl_multi.tftools.augmentation
import dl_multi.tftools.tflosses
import dl_multi.tftools.tfrecord
import dl_multi.tftools.tfsaver
import dl_multi.tftools.tfutils
import dl_multi.utils.general as glu

import tensorflow as tf

import dl_multi.models.tiramisu56
import dl_multi.archive.augmentation

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def train(
        param_info,
        param_log,
        param_batch,
        param_save, 
        param_train
    ): 
    
    _logger.info("Start training multi task classification and regression model with settings:\nparam_info:\t{}\nparam_log:\t{}\nparam_batch:\t{},\nparam_save:\t{},\nparam_train:\t{}".format(param_info, param_log, param_batch, param_save, param_train))

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    
    # Create the log and checkpoint folders if they do not exist
    checkpoint = dl_multi.utils.general.Folder().set_folder(**param_train["checkpoint"])
    log_dir = dl_multi.utils.general.Folder().set_folder(**param_log)

    tasks = len(param_train["objective"]) if isinstance(param_train["objective"], list) else 1
    
    data_io = dl_multi.tftools.tfrecord.tfrecord(param_train["tfrecord"], param_info, param_train["input"], param_train["output"])
    data = data_io.get_data()

    image_vaihingen = tf.to_float(data[0]) / 127.5 - 1.
    dsm_vaihingen =  tf.image.per_image_standardization(data[2])
    annotation_vaihingen = data[1]+1

    image_vaihingen, annotation_vaihingen, dsm_vaihingen = dl_multi.archive.augmentation.rnd_crop_rotate_90_with_flips_height(image_vaihingen, annotation_vaihingen, dsm_vaihingen, [224,224], 0.95, 1.1)
    
    #   execution -----------------------------------------------------------
    # ----------------------------------------------------------------------- 

    # Create batches by randomly shuffling tensors. The capacity specifies the maximum of elements in the queue
    image_batch_vaihingen, annotation_batch_vaihingen, dsm_batch_vaihingen = tf.train.shuffle_batch(
        [image_vaihingen, annotation_vaihingen, dsm_vaihingen],
        batch_size=2,
        capacity=64,
        min_after_dequeue=32,
        num_threads=16
    )

    with tf.compat.v1.variable_scope("net"):
        pred_vaihingen, reg_vaihingen = dl_multi.models.tiramisu56.multi_task_classification_regression(image_batch_vaihingen)

    # mask_vaihingen = tf.to_float(tf.squeeze(tf.greater(annotation_batch_vaihingen, 0.)))
    labels_vaihingen = tf.to_int32(tf.squeeze(tf.maximum(annotation_batch_vaihingen-1, 0), axis=3))

    pred_loss_vaihingen = tf.reduce_mean(tf.compat.v1.losses.compute_weighted_loss(
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_vaihingen, logits=pred_vaihingen)))

                
    reg_loss_vaihingen = tf.compat.v1.losses.mean_squared_error(dsm_batch_vaihingen, reg_vaihingen)

    task_weight = 0.5
    loss_vaihingen = task_weight * pred_loss_vaihingen + (1. - task_weight) * reg_loss_vaihingen

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.contrib.opt.AdamWOptimizer(0).minimize(loss_vaihingen)

    #   tfsession -----------------------------------------------------------
    # -----------------------------------------------------------------------
    
    # Operation for initializing the variables.
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                   tf.compat.v1.local_variables_initializer())                
    saver = dl_multi.tftools.tfsaver.Saver(tf.compat.v1.train.Saver(), **param_save, logger=_logger)
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in saver:
        
            loss_v, loss_v_r, _ = sess.run([ pred_loss_vaihingen, reg_loss_vaihingen, train_step])

            print(loss_v, loss_v_r)

            saver.save(sess, checkpoint, step=True)

        coord.request_stop()
        coord.join(threads)
        saver.save(sess, checkpoint)
    #   tfsession -----------------------------------------------------------
    # ----------------------------------------------------------------------- 
