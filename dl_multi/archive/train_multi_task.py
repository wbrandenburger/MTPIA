# ===========================================================================
#   train.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger 
import dl_multi.tftools.tfrecord
import dl_multi.tftools.augmentation
import dl_multi.tftools.tflosses

import os
import tensorflow as tf
import sys

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


    img, output_1, output_2 = dl_multi.tftools.tfrecord.read_tfrecord_queue(tf.train.string_input_producer([param_train["tfrecords"]]))

    img = dl_multi.plugin.get_module_task("tftools", param_train["input"]["method"], "tfnormalization")(img, **param_train["input"]["param"])
    output_1 = dl_multi.plugin.get_module_task("tftools", param_train["output"][1]["method"], "tfnormalization")(output_1, **param_train["output"][1]["param"])
    output_2 = dl_multi.plugin.get_module_task("tftools", param_train["output"][0]["method"], "tfnormalization")(output_2, **param_train["output"][0]["param"])

    img, output_1, output_2 = dl_multi.tftools.augmentation.rnd_crop_rotate_90_with_flips_height(img, output_1, output_2+1, param_train["image-size"], 0.95, 1.1)

    # Create batches by randomly shuffling tensors. The capacity specifies the maximum of elements in the queue
    img_batch, truth_batch, height_batch = tf.train.shuffle_batch(
        [img, output_2, output_1], **param_batch)

    losses = dl_multi.tftools.tflosses.Losses(param_train["objective"], logger=_logger)

    #   execution -----------------------------------------------------------
    # ----------------------------------------------------------------------- 
    with tf.variable_scope("net"):
        pred = dl_multi.plugin.get_module_task("models", *param_train["model"])(img_batch)

    pred_losses = losses.update([truth_batch, height_batch], list(pred))

    task_weight = 0.9
    loss = task_weight * pred_losses[0]  + (1. - task_weight) * pred_losses[2] 

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step_both = tf.contrib.opt.AdamWOptimizer(0).minimize(loss)
        
    # tf.summary.scalar('loss', loss)
    # tf.summary.scalar('accuracy', pred_losses[1])
    # tf.summary.scalar('pred_loss', pred_losses[0])
    # tf.summary.scalar('reg_loss', pred_losses[2])

    # merged_summary_op = tf.summary.merge_all()
    # summary_string_writer = tf.summary.FileWriter(log_dir)

    #   tfsession -----------------------------------------------------------
    # -----------------------------------------------------------------------
    # Operation for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())                
    saver = dl_multi.tftools.tfsaver.Saver(tf.train.Saver(), **param_save, logger=_logger
    )
    with tf.Session() as sess:
        sess.run(init_op)
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # iterate epochs
        for epoch in saver:
            result = sess.run([*pred_losses, #merged_summary_op,
                    train_step_both])
            print(losses.print_current_stats(epoch._index, result))
            # summary_string_writer.add_summary(summary_string, epoch._index)
            saver.save(sess, checkpoint, step=True)

        coord.request_stop()
        coord.join(threads)
        saver.save(sess, checkpoint)
    #   tfsession -----------------------------------------------------------
    # ----------------------------------------------------------------------- 
    # summary_string_writer.close()