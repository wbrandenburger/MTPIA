# ===========================================================================
#   eval_single -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger 
import dl_multi.tools.imgtools as imgtools 
import dl_multi.tools.patches
import dl_multi.plugin
import dl_multi.utils.time
import dl_multi.metrics.metrics

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
get_value = lambda obj, key, default: obj[key] if key in obj.keys() else default

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
    img_set, save = dl_multi.tools.data.get_data(files, **output, specs=specs, param_label=param_label)

    # Create the log and checkpoint folders if they do not exist
    folder = dl_multi.utils.general.Folder()
    checkpoint = folder.set_folder(
        param_eval["checkpoints"], name=[param_eval["checkpoint"]]
    )
    log_file = folder.set_folder(
        param_eval["logs"], name=[param_eval["checkpoint"] + ".eval.log"]
    )

    eval_obj = dl_multi.metrics.metrics.Metrics(
        param_eval["objective"], 
        len(img_set), 
        tasks=param_eval["tasks"], 
        categories=len(param_label), 
        labels=list(param_label.values()), 
        label_spec=param_class,
        sklearn=get_value(param_eval, "sklearn", True),
        logger=_logger
    )

    time_obj_img = dl_multi.utils.time.MTime(number=len(img_set), label="IMAGE")

    #   execution -----------------------------------------------------------
    # -----------------------------------------------------------------------
    for item, time_img, eval_img in zip(img_set, time_obj_img, eval_obj):
        img = item.spec("image").data
        truth = [item.spec(param_eval["truth"][task]).data for task in range(param_eval["tasks"])]

        patches = dl_multi.tools.patches.Patches(
            img,
            tasks=param_eval["tasks"],
            obj=param_eval["objective"],
            categories=len(param_label), 
            limit=param_eval["limit"], 
            margin=param_eval["margin"], 
            pad=param_eval["pad"], 
            stitch=param_eval["stitch"],
            logger = _logger
        )

        for patch in patches:
            patch.status()

            tf.reset_default_graph()
            tf.Graph().as_default()
            
            input_norm = dl_multi.plugin.get_module_task("tftools", param_eval["input-norm"], "tfnormalization" )          
            data = tf.expand_dims(input_norm(patch.get_image_patch()), 0)
      
            with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
                pred = dl_multi.plugin.get_module_task("models", *param_eval["model"])(data)

            # Operation for initializing the variables.
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            #   tfsession ---------------------------------------------------
            # ---------------------------------------------------------------
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
            #   tfsession ---------------------------------------------------
            # ---------------------------------------------------------------
            patch.time()    

        label = item.spec(get_value(param_eval, "truth_label", None)).data if get_value(param_eval, "truth_label", None) else None
        eval_img.update(
            truth, 
            [patches.get_img(task=task) for task in range(param_eval["tasks"])],
            label=label
        )
        print(eval_img.print_current_stats())

        for task in range(param_eval["tasks"]):
            save(item.spec(param_eval["truth"][task]).path, patches.get_img(task=task), index=param_eval["truth"][task])
            # _logger.debug("Result with {}".format(imgtools.get_img_information(patches.get_img(task=task))))
        
        time_img.stop()
        _logger.debug(time_img.overall())
        _logger.debug(time_img.stats())
    
    print(eval_obj)
    eval_obj.write_log(log_file, write="w+", verbose=True)             