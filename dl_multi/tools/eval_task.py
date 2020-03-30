# ===========================================================================
#   eval_single -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger 
import dl_multi.tools.imgtools as imgtools 
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
    param_eval,
    param_label, 
    param_class
    ):

    _logger.debug("Start training single task classifciation model with settings:\noutput:\t{}\nparam_eval:\t{}\nparam_label:\t{}\nparam_class:\t{}".format(output, param_eval, param_label, param_class))   

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    img_set, save = dl_multi.tools.data.get_data(files, **output, specs=specs)

    checkpoint = param_eval["checkpoints"] + "\\" + param_eval["checkpoint"]
    logfile = param_eval["logs"] + "\\" + param_eval["checkpoint"] + ".eval.log"

    # eval_obj = dl_multi.tools.evaluation.EvalReg(log=logfile) # change

    time_obj_img = dl_multi.utils.time.MTime(number=len(img_set), label="IMAGE")
    truth_spec = param_eval["truth"] if isinstance(param_eval["truth"], list) else [param_eval["truth"]]
    for item, time_img in zip(img_set, time_obj_img):
        img = item.spec("image").data

        patches = dl_multi.tools.patches.Patches(
            img,
            tasks=param_eval["tasks"],
            obj=param_eval["objective"],
            categories=len(param_label), 
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
                if param_eval["tasks"]==1: 
                    if param_eval["objective"]=="regression":
                        pred = dl_multi.tools.tiramisu56.tiramisu56_dsm(data)
                    else:
                        pred = dl_multi.tools.tiramisu56.tiramisu56_sem(data)
                else:
                    pred = dl_multi.tools.tiramisu56.tiramisu56(data)

            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init_op)
                saver.restore(sess, checkpoint)
                sess.graph.finalize()
                
                model_out = sess.run([pred])
                if param_eval["tasks"]==1: 
                    if param_eval["objective"]=="regression":
                        patch.set_patch([model_out[0]])
                    else:
                        patch.set_patch([model_out[0][0]])
                else:
                    patch.set_patch([model_out[0][0], model_out[0][2]])
                #eval_obj.update(patches.get_img(), truth)

            patch.time()     
            #print(eval_obj)
            
        for task in range(param_eval["tasks"]):
            save(item.spec(truth_spec[task]).path, patches.get_img(task=task), index=None if param_eval["tasks"]==1 else truth_spec[task])
        # _logger.debug("Result with {}".format(imgtools.get_img_information(patches.img)))
        
        time_img.stop()
        _logger.debug(time_img.overall())
        _logger.debug(time_img.stats())

    # eval_obj.write_log()         