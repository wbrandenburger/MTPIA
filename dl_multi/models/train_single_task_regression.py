# ===========================================================================
#   train.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger 
import dl_multi.tftools.tfaugmentation
import dl_multi.tftools.tflosses
import dl_multi.tftools.tfrecord
import dl_multi.tftools.tfsaver
import dl_multi.tftools.tfutils
import dl_multi.utils.general as glu

import tensorflow as tf

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def train(
        param_info,
        param_log,
        param_batch,
        param_save, 
        param_train
    ): 
    
    _logger.info("Start training multi task classification and regression model with settings:\nparam_info:\t{}\nparam_log:\t{}\nparam_batch:\t{},\nparam_save:\t{},\nparam_train:\t{}".format( param_info, param_log, param_batch, param_save, param_train))

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    
    # Create the log and checkpoint folders if they do not exist
    checkpoint = dl_multi.utils.general.Folder().set_folder(**param_train["checkpoint"])
    log_dir = dl_multi.utils.general.Folder().set_folder(**param_log)

    tasks = len(param_train["objective"]) if isinstance(param_train["objective"], list) else 1

    data_io = dl_multi.tftools.tfrecord.tfrecord(param_train["tfrecord"], param_info, param_train["input"], param_train["output"])
    data = data_io.get_data()
    data = dl_multi.tftools.tfutils.preprocessing(data, param_train["input"], param_train["output"])
    data = dl_multi.tftools.tfaugmentation.rnd_crop(data, param_train["image-size"], data_io.get_spec_item_list("channels"), data_io.get_spec_item_list("scale"), **param_train["augmentation"])

    objectives = dl_multi.tftools.tflosses.Losses(param_train["objective"], logger=_logger, **glu.get_value(param_train, "multi-task", dict()))

    #   execution -----------------------------------------------------------
    # -----------------------------------------------------------------------

    # Create batches by randomly shuffling tensors. The capacity specifies the maximum of elements in the queue
    data_batch = tf.train.shuffle_batch(data, **param_batch)
   
    input_batch = data_batch[0]
    output_batch = data_batch[1:] if isinstance(data_batch[1:], list) else [data_batch[1:]]

    with tf.compat.v1.variable_scope("net"):
        pred = dl_multi.plugin.get_module_task("models", *param_train["model"])(input_batch)
        pred = list(pred) if isinstance(pred, tuple) else [pred]

    objectives.update(output_batch, pred)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.contrib.opt.AdamWOptimizer(0).minimize(objectives.get_loss())

    #   tfsession -----------------------------------------------------------
    # -----------------------------------------------------------------------

    # Operation for initializing the variables.
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                   tf.compat.v1.local_variables_initializer())              
    saver = dl_multi.tftools.tfsaver.Saver(**param_save, logger=_logger)
    with tf.compat.v1.Session() as sess:
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
