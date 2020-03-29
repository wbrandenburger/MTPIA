# ===========================================================================
#   eval_single -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.tools.imgtools
import dl_multi.tools.patches
import dl_multi.tools.evaluation
import dl_multi.tools.tiramisu56
import dl_multi.utils.time

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
    param_eval
    ): 
    
    dl_multi.__init__._logger.debug("Start training single task regression model with settings:\noutput:\t{}\nparam:\t{}\nparam_label:\t{}".format(output, param, param_eval))

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    img_set, save = dl_multi.tools.data.get_data(files, **output, specs=specs)

    checkpoint = param_eval["checkpoints"] + "\\" + param_eval["checkpoint"]
    logfile = param_eval["logs"] + "\\" + param_eval["checkpoint"] + ".eval.log"

    eval_obj = dl_multi.tools.evaluation.EvalReg(log=logfile) # change
    time_obj_img = dl_multi.utils.time.MTime(len(img_set))

    for item, time_img in zip(img_set, time_obj_img):
        img = item.spec("image").data
        truth = item.spec(param_eval["truth"]).data

        dl_multi.__init__._logger.debug("Image with {}".format(dl_multi.tools.imgtools.get_img_information(img)))
        dl_multi.__init__._logger.debug("Truth with {}".format(dl_multi.tools.imgtools.get_img_information(truth)))

        patches = dl_multi.tools.patches.Patches(img, obj=param_eval["objective"], limit=param["limit"], margin=param["margin"], pad=param["pad"], stitch=param_eval["stitch"]) # change
        time_obj_patch = dl_multi.utils.time.MTime(len(patches))

        for patch, time_patch in zip(patches, time_obj_patch):
            patch.print_iter()

            tf.reset_default_graph()
            tf.Graph().as_default()
                    
            data = tf.cast(patch.get_image_patch() / 127.5 - 1., tf.float32)
            data = tf.expand_dims(data, 0)
      
            with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
                pred = dl_multi.tools.tiramisu56.tiramisu56_dsm(data) # change
      
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init_op)
                saver.restore(sess, checkpoint)
                sess.graph.finalize()
                
                model_out = sess.run([pred])
                patch.set_patch(model_out[0]) # change
                eval_obj.update(patches.img, truth)
            
            time_patch.stop(show=True)

        save(item.spec(param_eval["truth"]).path, patches.img)
        dl_multi.__init__._logger.debug("Result with {}".format(dl_multi.tools.imgtools.get_img_information(patches.img)))
        
        time_img.stop(show=True)
    # eval_obj.write_log()         