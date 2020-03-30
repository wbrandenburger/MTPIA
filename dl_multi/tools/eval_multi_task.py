# ===========================================================================
#   eval_single -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger 
import dl_multi.tools.imgtools as imgtools 
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
    param_eval, 
    param_label,
    param_class
    ): 
    
    _logger.debug("Start training multi task classifciation and regression model with settings:\noutput:\t{}\nparam_eval:\t{}\nparam_label:\t{}\nparam_class:\t{}".format(output, param_eval, param_label, param_class))  

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    img_set, save = dl_multi.tools.data.get_data(files, **output, specs=specs, param_label=param_label)

    checkpoint = param_eval["checkpoints"] + "\\" + param_eval["checkpoint"]
    logfile = param_eval["logs"] + "\\" + param_eval["checkpoint"] + ".eval.log"

    # eval_obj = dl_multi.tools.evaluation.EvalCat(param_label, param_class, log=logfile) # change

    time_obj_img = dl_multi.utils.time.MTime(number=len(img_set), label="IMAGE")
    for item, time_img in zip(img_set, time_obj_img):
        img = item.spec("image").data
        truth_task_a = item.spec(param_eval["truth"][0]).data
        truth_task_b = item.spec(param_eval["truth"][1]).data

        patches = dl_multi.tools.patches.Patches(
            img, 
            tasks=2,
            obj=param_eval["objective"], 
            categories=[len(param_label), 1], 
            limit=param_eval["limit"], 
            margin=param_eval["margin"], 
            pad=param_eval["pad"], 
            stitch=param_eval["stitch"],
            log = _logger
        ) 
        for patch in patches:
            patch.status()

            tf.reset_default_graph()
            tf.Graph().as_default()
                    
            data = tf.cast(patch.get_image_patch() / 127.5 - 1., tf.float32)
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
                patch.set_patch([model_out[0][0], model_out[0][2]]) # change

            patch.time()    

        save(item.spec(param_eval["truth"][0]).path, patches.get_img(task=0), index=param_eval["truth"][0])
        # _logger.debug("Result with {}".format(imgtools.get_img_information(patch_task_a.img)))

        save(item.spec(param_eval["truth"][1]).path, patches.get_img(task=1), index=param_eval["truth"][1])
        # _logger.debug("Result with {}".format(imgtools.get_img_information(patch_task_b.img)))
        
        time_img.stop()
        _logger.debug(time_img.overall())
        _logger.debug(time_img.stats())
# eval_obj.write_log() 
        
