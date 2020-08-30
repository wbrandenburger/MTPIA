# ===========================================================================
#   eval_single -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger 
import dl_multi.metrics.metrics
import dl_multi.plugin
import dl_multi.utils.patches
import dl_multi.utils.general as glu
import dl_multi.utils.imgio
from dl_multi.utils import imgtools as imgtools 
import dl_multi.utils.time

import tensorflow as tf

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def eval(
    files,
    param_specs,
    param_io,
    param_log,
    param_ckpt, 
    param_eval, 
    param_label,
    param_classes
    ): 
    
    _logger.info("Start training multi task classification and regression model with settings:\nparam_io:\t{}\nparam_log:\t{}\nparam_ckpt:\t{}\nparam_eval:\t{}\nparam_label:\t{}\nparam_class:\t{}".format(param_io, param_log, param_ckpt, param_eval, param_label, param_classes))  

    #   settings ------------------------------------------------------------
    # -----------------------------------------------------------------------
    param_class = [c["name"] for c in param_classes]

    # Create the log and checkpoint folders if they do not exist
    param_io_cp = param_io.copy()
    param_ckpt_cp = param_ckpt.copy()
    param_log_cp = param_log.copy()
    
    ckpts = param_ckpt["list"] if "list" in param_ckpt.keys() else range(*param_ckpt["range"])
    for ckpt in ckpts:
        format_ckpt = {"ckpt": ckpt, "plch": "{}"}
        param_io_cp["path_name"] = param_io["path_name"].format(**format_ckpt)
        param_ckpt_cp["path_name"] = param_ckpt["path_name"].format(**format_ckpt)
        param_log_cp["path_name"] = param_log["path_name"].format(**format_ckpt)

        img_in, img_out, log_out, _ = dl_multi.utils.imgio.get_data(files, param_specs, param_io_cp, param_log=param_log, param_label=param_label)

        checkpoint = glu.Folder().set_folder(**param_ckpt_cp)
        log_file = glu.Folder().set_folder(**param_log_cp)


        tasks = len(param_eval["objective"]) if isinstance(param_eval["objective"], list) else 1
        print(tasks)

        eval_obj = dl_multi.metrics.metrics.Metrics(
            param_eval["objective"], 
            len(img_in), 
            categories=len(param_label), 
            labels=list(param_label.values()), 
            label_spec=param_class,
            sklearn=glu.get_value(param_eval, "sklearn", True),
            logger=_logger
        )

        time_obj_img = dl_multi.utils.time.MTime(number=len(img_in), label="IMAGE")
        
        #   execution -----------------------------------------------------------
        # -----------------------------------------------------------------------
        for item, time_img, eval_img in zip(img_in, time_obj_img, eval_obj):
            img = dl_multi.plugin.get_module_task("tftools", param_eval["input"]["method"], "normalization")(item.spec("image").data, **param_eval["input"]["param"])
            truth = [dl_multi.plugin.get_module_task("tftools", param_eval["output"][task]["method"], "normalization")(imgtools.expand_image_dim(item.spec(param_eval["truth"][task]).data, **param_eval["output"][task]["param"])) for task in range(tasks)]
            # print(type(truth), len(truth))
            # print(truth[0])

            patches = dl_multi.utils.patches.Patches(
                img,
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
        
                data = tf.expand_dims(patch.get_image_patch(), 0)
        
                with tf.compat.v1.variable_scope("net", reuse=tf.AUTO_REUSE):
                    pred = dl_multi.plugin.get_module_task("models", *param_eval["model"])(data)
                #   tfsession ---------------------------------------------------
                # ---------------------------------------------------------------
                # Operation for initializing the variables.
                init_op = tf.compat.v1.global_variables_initializer()
                saver = tf.compat.v1.train.Saver()
                with tf.compat.v1.Session() as sess:
                    sess.run(init_op)
                    saver.restore(sess, checkpoint)
                    sess.graph.finalize()
                    
                    model_out = sess.run([pred])
                    patch.set_patch([model_out[0][0]])
                patch.time()
                #   tfsession ---------------------------------------------------
                # ---------------------------------------------------------------

        #   output --------------------------------------------------------------
        # -----------------------------------------------------------------------
            label = item.spec(glu.get_value(param_eval, "truth_label", None)).data if glu.get_value(param_eval, "truth_label", None) else None
        
            for task in range(tasks):
                img_out(item.spec(param_eval["truth"][task]).path, patches.get_img(task=task), prefix=param_eval["truth"][task])
            
            eval_img.update(truth, [patches.get_img(task=task) for task in range(tasks)],
                label=label
            )
            eval_obj.write_log([log_out(item.spec(param_eval["truth"][task]).log,  prefix=param_eval["truth"][task]) for task in range(tasks)], write="w+", current=True, verbose=True)
            print(eval_img.print_current_stats())
            
            time_img.stop()
            _logger.info(time_img.overall())
            _logger.info(time_img.stats())
    
        eval_obj.write_log(log_file, verbose=True)
        # print(eval_obj)   