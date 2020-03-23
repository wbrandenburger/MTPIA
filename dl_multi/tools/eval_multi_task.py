# ===========================================================================
#   eval_single -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.tools.imgtools
import dl_multi.tools.patches
import dl_multi.tools.evaluation
import dl_multi.tools.tiramisu56

import logging
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import os
import tensorflow as tf
import tifffile

from PIL import Image

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def eval(
    files,
    specs,
    output,    
    param,
    param_eval, 
    param_label,
    param_class
    ): 
    
    dl_multi.__init__._logger.debug("Start training multi task classifciation and regression model with settings:\noutput:\t{}\nparam:\t{}\nparam_eval:\t{}\nparam_label:\t{}\nparam_class:\t{}".format(output, param, param_eval, param_label, param_class))  

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    img_set, save = dl_multi.tools.data.get_data(files, **output, specs=specs, param_label=param_label)

    checkpoint = param_eval["checkpoints"] + "\\" + param_eval["checkpoint"]
    logfile = param_eval["logs"] + "\\" + param_eval["checkpoint"] + ".eval.log"

    for item in img_set:

        img = item.spec("image").data
        truth_task_a = item.spec(param_eval["truth"][0]).data
        truth_task_b = item.spec(param_eval["truth"][1]).data

        patches_task_a = dl_multi.tools.patches.Patches(img, obj=param_eval["objective"][0], categories=len(param_label), limit=param["limit"], margin=param["margin"], pad=param["pad"]) 

        patches_task_b = dl_multi.tools.patches.Patches(img, obj=param_eval["objective"][1], limit=param["limit"], margin=param["margin"], pad=param["pad"])

        for patch_task_a, patch_task_b in zip(patches_task_a, patches_task_b):
            patch_task_a.print_iter()

            tf.reset_default_graph()
            tf.Graph().as_default()
                    
            data = tf.cast(patch_task_a.get_image_patch() / 127.5 - 1., tf.float32)
            patch_task_b.get_image_patch()

            data = tf.expand_dims(data, 0)
      
            with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
                pred = dl_multi.tools.tiramisu56.tiramisu56(data) # change
      
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init_op)
                saver.restore(sess, checkpoint)
                sess.graph.finalize()
                
                model_out = sess.run([pred])
                patch_task_a.set_patch(model_out[0][0]) # change
                patch_task_b.set_patch(model_out[0][2]) # change

        save(item.spec(param_eval["truth"][0]).path, patch_task_a.img, index=param_eval["truth"][0])
        dl_multi.__init__._logger.debug("Result with {}".format(dl_multi.tools.imgtools.get_img_information(patch_task_a.img)))

        save(item.spec(param_eval["truth"][1]).path, patch_task_b.img, index=param_eval["truth"][1])
        dl_multi.__init__._logger.debug("Result with {}".format(dl_multi.tools.imgtools.get_img_information(patch_task_b.img)))
        
