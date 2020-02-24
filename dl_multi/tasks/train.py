# ===========================================================================
#   train_decomposition.py --------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.config.settings
import dl_multi.utils.format

import tensorflow as tf
import numpy as np
import os, sys
import matplotlib.pyplot as plt

#   settings ----------------------------------------------------------------
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def task_default():

    # print user defined settings
    dl_multi.__init__._logger.debug("Print user defined settings")
    dl_multi.utils.format.print_data(dl_multi.config.settings._SETTINGS)
