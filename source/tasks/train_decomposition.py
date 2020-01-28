# ===========================================================================
#   train_decomposition.py --------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import __init__
import config.settings
import utils.format

import tensorflow as tf
import numpy as np
import os, sys
import matplotlib.pyplot as plt

#   settings ----------------------------------------------------------------
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():
    # print user defined settings
    __init__._logger.debug("Print user defined settings")
    utils.format.print_data(config.settings._SETTINGS)
