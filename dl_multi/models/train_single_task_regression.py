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

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def train(
        param_specs,
        param_info,
        param_log,
        param_batch,
        param_save, 
        param_train
    ): 
    
    _logger.debug("Start training multi task classification and regression model with settings:\nparam_specs:\t{}\nparam_info:\t{}\nparam_log:\t{}\nparam_batch:\t{},\nparam_save:\t{},\nparam_train:\t{}".format(param_specs, param_info, param_log, param_batch, param_save, param_train))

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    
    # Create the log and checkpoint folders if they do not exist
    checkpoint = dl_multi.utils.general.Folder().set_folder(**param_train["checkpoint"])
    log_dir = dl_multi.utils.general.Folder().set_folder(**param_log)

    tasks = len(param_train["objective"]) if isinstance(param_train["objective"], list) else 1

    data = dl_multi.tftools.tfrecord.get_data(param_train["tfrecord"], param_specs, param_info, param_train["input"], param_train["output"])
    data = dl_multi.tftools.tfutils.preprocessing(data, param_train["input"], param_train["output"])

    img, truth, _ = dl_multi.tftools.augmentation.rnd_crop_rotate_90_with_flips_height(data[0], data[1], data[1], param_train["image-size"], 0.95, 1.1)

    objectives = dl_multi.tftools.tflosses.Losses(param_train["objective"], logger=_logger, **glu.get_value(param_train, "multi-task", dict()))

    #   execution -----------------------------------------------------------
    # -----------------------------------------------------------------------

    # Create batches by randomly shuffling tensors. The capacity specifies the maximum of elements in the queue
    img_batch, truth_batch = tf.train.shuffle_batch(
        [img, truth], **param_batch)

    with tf.variable_scope("net"):
        pred = dl_multi.plugin.get_module_task("models", *param_train["model"])(img_batch)
    objectives.update([truth_batch], list(pred))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.contrib.opt.AdamWOptimizer(0).minimize(objectives.get_loss())

    #   tfsession -----------------------------------------------------------
    # -----------------------------------------------------------------------

    # Operation for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())              
    saver = dl_multi.tftools.tfsaver.Saver(tf.train.Saver(), **param_save, logger=_logger)
    with tf.Session() as sess:
        sess.run(init_op)
            
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Iteration over epochs        
        for epoch in saver:
            stats_epoch, _ = sess.run([objectives.get_stats(), train_step])
            print(objectives.get_stats_str(epoch._index, stats_epoch))
            saver.save(sess, checkpoint, step=True)

        coord.request_stop()
        coord.join(threads)
        saver.save(sess, checkpoint)
    #   tfsession -----------------------------------------------------------
    # -----------------------------------------------------------------------
