# ===========================================================================
#   dl_multi.py -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__

import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
get_value = lambda obj, key, default: obj[key] if key in obj.keys() else default

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def set_cuda_properties(param):
    if not param:
        return
    
    set_cuda_visible_devices(get_value(param, "visible_devices", None))

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def set_cuda_visible_devices(devices):
    if not devices:
        return
    
    os.environ["CUDA_VISIBLE_DEVICES"]=devices