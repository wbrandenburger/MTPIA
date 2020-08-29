# ===========================================================================
#   train.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger 
import dl_multi.tftools.tfrecord
import dl_multi.tftools.augmentation

import os
import tensorflow as tf

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def train(
        param_log,
        param_batch,
        param_save, 
        param_train
    ): 
    
    _logger.debug("Start training multi task classification and regression model with settings:\n'param_log':\t'{}'\n'param_batch':\t'{}',\n'param_save':\t'{}',\n'param_train':\t'{}'".format(param_log, param_batch, param_save,param_train))

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------

    # Create the log and checkpoint folders if they do not exist
    folder = dl_multi.utils.general.Folder()
    checkpoint = folder.set_folder(**param_train["checkpoint"])
    log_dir = folder.set_folder(**param_log)

    img, PLACEHOLDER, truth = dl_multi.tftools.tfrecord.read_tfrecord_queue(tf.train.string_input_producer([param_train["tfrecords"]]))

    img = dl_multi.plugin.get_module_task("tftools", param_train["input"]["method"], "tfnormalization")(img, **param_train["input"]["param"])
    truth = dl_multi.plugin.get_module_task("tftools", param_train["output"]["method"], "tfnormalization")(PLACEHOLDER, **param_train["output"]["param"])

    img, _, truth = dl_multi.tftools.augmentation.rnd_crop_rotate_90_with_flips_height(img, PLACEHOLDER, truth + 1, param_train["image-size"], 0.95, 1.1)

    # Create batches by randomly shuffling tensors. The capacity specifies the maximum of elements in the queue
    img_batch,truth_batch = tf.train.shuffle_batch(
        [img, truth], **param_batch)

    #   execution -----------------------------------------------------------
    # ----------------------------------------------------------------------- 
    with tf.compat.v1.variable_scope("net"):
        pred, argmax = dl_multi.plugin.get_module_task("models", *param_train["model"])(img_batch)

    mask = tf.to_float(tf.squeeze(tf.greater(truth_batch, 0.)))
    truth = tf.to_int32(tf.squeeze(tf.maximum(truth_batch-1, 0), axis=3))

    loss = tf.reduce_mean(
        tf.compat.v1.losses.compute_weighted_loss(
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=truth, 
                logits=pred
            ),
            weights = mask
        )
    )

    acc = 1 - ( tf.count_nonzero((tf.to_float(argmax)-tf.to_float(truth)), dtype=tf.float32)
            / (param_train["image-size"][0] * param_train["image-size"][1] * param_batch["batch_size"]))

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step_both = tf.contrib.opt.AdamWOptimizer(0).minimize(loss)
        
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)
    merged_summary_op = tf.summary.merge_all()
    summary_string_writer = tf.summary.FileWriter(log_dir)

    #   tfsession -----------------------------------------------------------
    # -----------------------------------------------------------------------
    # The op for initializing the variables.
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                   tf.compat.v1.local_variables_initializer()) 
    saver = dl_multi.tftools.tfsaver.Saver(tf.compat.v1.train.Saver(), **param_save, logger=_logger)
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
            
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
            
        loss_v = 0
        acc_v = 0
        loss_v_r = 0

        # iterate epochs
        for epoch in saver:
            loss_v, acc_v, summary_string, _ = sess.run([loss, acc, merged_summary_op, train_step_both])
            
            summary_string_writer.add_summary(summary_string, epoch._index)
                
            print("Step: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch._index, loss_v, acc_v))
            saver.save(sess, checkpoint, step=True)

        coord.request_stop()
        coord.join(threads)
        saver.save(sess, checkpoint)
    #   tfsession -----------------------------------------------------------
    # -----------------------------------------------------------------------
    summary_string_writer.close()