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
import sys

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
get_value = lambda obj, key, default: obj[key] if key in obj.keys() else default

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

    input_norm = dl_multi.plugin.get_module_task("tftools", param_train["input-norm"], "tfnormalization")  

    img, output_1, output_2 = dl_multi.tftools.tfrecord.read_tfrecord_queue(tf.train.string_input_producer([param_train["tfrecords"]]))

    img = dl_multi.plugin.get_module_task("tftools", param_train["input"]["method"], "tfnormalization")(img, **param_train["input"]["param"])
    output_1 = dl_multi.plugin.get_module_task("tftools", param_train["output"][1]["method"], "tfnormalization")(output_1, **param_train["output"][1]["param"])
    output_2 = dl_multi.plugin.get_module_task("tftools", param_train["output"][0]["method"], "tfnormalization")(output_2, **param_train["output"][0]["param"])

    img, output_1, output_2 = dl_multi.tftools.augmentation.rnd_crop_rotate_90_with_flips_height(img, output_1, output_2+1, param_train["image-size"], 0.95, 1.1)

    # Create batches by randomly shuffling tensors. The capacity specifies the maximum of elements in the queue
    img_batch, label_batch, height_batch = tf.train.shuffle_batch(
        [img, output_2, output_1], **param_batch)

    #   execution -----------------------------------------------------------
    # ----------------------------------------------------------------------- 
    with tf.variable_scope("net"):
        pred, argmax, reg = dl_multi.plugin.get_module_task("models", *param_train["model"])(img_batch)

    mask = tf.to_float(tf.squeeze(tf.greater(label_batch, 0.)))
    labels = tf.to_int32(tf.squeeze(tf.maximum(label_batch-1, 0), axis=3))
    
    pred_loss = tf.reduce_mean(
        tf.losses.compute_weighted_loss(
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, 
                logits=pred
            ),
            weights = tf.to_float(mask)
        )
    )

    acc = 1 - ( tf.count_nonzero((tf.to_float(argmax)-tf.to_float(labels)), dtype=tf.float32)
            / (param_train["image-size"][0] * param_train["image-size"][1] * param_batch["batch_size"]))
                
    reg_loss= tf.losses.mean_squared_error(height_batch, reg, weights = tf.expand_dims(mask, axis=3))


    task_weight = 0.9
    loss = task_weight * pred_loss + (1. - task_weight) * reg_loss

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step_both = tf.contrib.opt.AdamWOptimizer(0).minimize(loss)
        
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('pred_loss', pred_loss)
    tf.summary.scalar('reg_loss', reg_loss)

    merged_summary_op = tf.summary.merge_all()
    summary_string_writer = tf.summary.FileWriter(log_dir)

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
            
        loss_v = 0
        acc_v = 0
        loss_v_r = 0

        # iterate epochs
        for epoch in saver:
            loss_v, loss_v_r, acc_v, summary_string, _ = sess.run([ pred_loss, reg_loss, acc, merged_summary_op, train_step_both])
            
            summary_string_writer.add_summary(summary_string, epoch._index)
                
            print("Step: {}, Loss_CLS: {:.3f}, Accuracy: {:.3f}, Loss_REG: {:.3f}".format(epoch._index, loss_v, acc_v, loss_v_r))

            saver.save(sess, checkpoint, step=True)

        coord.request_stop()
        coord.join(threads)
        saver.save(sess, checkpoint)
    #   tfsession -----------------------------------------------------------
    # ----------------------------------------------------------------------- 
    summary_string_writer.close()