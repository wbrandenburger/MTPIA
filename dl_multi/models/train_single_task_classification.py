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
get_value = lambda obj, key, default: obj[key] if key in obj.keys() else default

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def train(
    param_train
    ): 
    
    _logger.debug("Start training single task classification model with settings:\n'param_train':\t'{}'".format(param_train))

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------

    # Create the log and checkpoint folders if they do not exist
    folder = dl_multi.utils.general.Folder()
    checkpoint = folder.set_folder(
        param_train["checkpoints"], name=[param_train["checkpoint"]]
    )
    log_dir = folder.set_folder(param_train["logs"])

    img, height, label = dl_multi.tftools.tfrecord.read_tfrecord_queue(tf.train.string_input_producer([param_train["tfrecords"]]))
    input_norm = dl_multi.plugin.get_module_task("tftools", param_train["input-norm"], "tfnormalization" )          
    img = input_norm(img)

    img, height, label = dl_multi.tftools.augmentation.rnd_crop_rotate_90_with_flips_height(img, height, label + 1, param_train["image-size"], 0.95, 1.1)

    # Create batches by randomly shuffling tensors. The capacity specifies the maximum of elements in the queue
    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label], **param_train["batch"])

    #   execution -----------------------------------------------------------
    # ----------------------------------------------------------------------- 
    with tf.variable_scope("net"):
        pred, argmax = dl_multi.plugin.get_module_task("models", *param_train["model"])(img_batch)

    mask = tf.to_float(tf.squeeze(tf.greater(label_batch, 0.)))
    labels = tf.to_int32(tf.squeeze(tf.maximum(label_batch-1, 0), axis=3))

    loss = tf.reduce_mean(
        tf.losses.compute_weighted_loss(
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, 
                logits=pred
            ),
        weights = tf.to_float(mask)
        )
    )

    acc = 1 - ( tf.count_nonzero((tf.to_float(argmax)-tf.to_float(labels)), dtype=tf.float32)
            / (param_train["image-size"][0] * param_train["image-size"][1] * param_train["batch"]["batch_size"]))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step_both = tf.contrib.opt.AdamWOptimizer(0).minimize(loss)
        
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)
    merged_summary_op = tf.summary.merge_all()
    summary_string_writer = tf.summary.FileWriter(log_dir)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer()) 

    saver = dl_multi.tftools.tfsaver.Saver(
        tf.train.Saver(), **param_train["tfsave"], logger=_logger
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
            loss_v, acc_v, summary_string, _ = sess.run([loss, acc, merged_summary_op, train_step_both])
            
            summary_string_writer.add_summary(summary_string, epoch._index)
                
            print("Step: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch._index, loss_v, acc_v))
            saver.save(sess, checkpoint, step=True)

        coord.request_stop()
        coord.join(threads)
        saver.save(sess, checkpoint)

    summary_string_writer.close()