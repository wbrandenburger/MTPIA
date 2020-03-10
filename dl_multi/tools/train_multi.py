# ===========================================================================
#   train.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.tools.tfrecord
import dl_multi.tools.augmentation
import dl_multi.tools.tiramisu56

import os
import tensorflow as tf

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def train(param_train, param_out): 
    
    dl_multi.__init__._logger.debug("Start training single task model with settings:\n'param_train':\t'{}',\n'param_out':\t'{}'".format(param_train, param_out))

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    batch_size = param_train["batch-size"]
    num_epochs = param_train["num-epochs"] 
    img_size = param_train["image-size"]
    
    tfrecords_queue = tf.train.string_input_producer([param_out["tfrecords"]])
    checkpoints_dir = param_out["checkpoints"]
    log_dir = param_out["logs"]
    # if not os.path.isdir(output_dir): os.mkdir(output_dir)

    img, height, label = dl_multi.tools.tfrecord.read_tfrecord_queue(tfrecords_queue)
    img = tf.to_float(img)
    img, height, label = dl_multi.tools.augmentation.rnd_crop_rotate_90_with_flips_dsm(img, height, label + 1, img_size, 0.95, 1.1)
    
    img = img / 127.5 - 1.
    #height = tf.image.per_image_standardization(height)
    # Create batches of 'batch size'  images, labels and dsm by randomly shuffling tensors. The capacity specifies the maximum of elements in the queue
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/shuffle_batch#for_example
    img_batch, label_batch, dsm_batch = tf.train.shuffle_batch(
        [img, label, height],
        batch_size=batch_size,
        capacity=16, # 64
        min_after_dequeue=8,
        num_threads=4 #16
    )

    with tf.variable_scope("net"):
        pred, argmax, reg = dl_multi.tools.tiramisu56.tiramisu56(img_batch)

    mask= tf.to_float(tf.squeeze(tf.greater(label_batch, 0.)))
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
            / (img_size[0] * img_size[1] * batch_size))
                
    reg_loss= tf.losses.mean_squared_error(dsm_batch, reg, weights = tf.expand_dims(mask, axis=3))


    task_weight = 0.5
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

    # Create the log folder if doesn't exist log_folderyet
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
                    
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init_op)
            
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
            

        loss_v = 0
        acc_v = 0
        loss_v_r = 0


        # iterate epochs (eigentlich iterations!!!)
        for i in range(num_epochs+1):

            loss_v, loss_v_r, acc_v, summary_string, _ = sess.run([ pred_loss, reg_loss, acc, merged_summary_op, train_step_both])
            
            summary_string_writer.add_summary(summary_string, i)
                
            print("Step: " + str(i) + " Loss_PRED: " + str(loss_v) + " Loss_REG: " + str(loss_v_r) + " Accuracy_V: " + str(acc_v) )
                
            if i % 100 == 0:
                save_path = saver.save(sess, checkpoints_dir + "\\pia.ckpt")
                print("Model saved in file: %s" % save_path)
            if i % 25000 == 0:
                save_path = saver.save(sess, checkpoints_dir + "\\pia.ckpt", global_step=i)
                print("Model saved in file: %s" % save_path)


        coord.request_stop()
        coord.join(threads)
        
        save_path = saver.save(sess, checkpoints_dir+"\\pia.ckpt", global_step=i)
        print("Model saved in file: %s" % save_path)
        
    summary_string_writer.close()