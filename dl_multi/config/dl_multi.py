# ===========================================================================
#   dl_multi.py -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.utils.general as glu

import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def set_cuda_properties(param):
    if not param:
        return
    
    set_cuda_visible_devices(glu.get_value(param, "visible_devices", None))

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def set_cuda_visible_devices(devices):
    if not devices:
        return
    
    os.environ["CUDA_VISIBLE_DEVICES"]=devices