# ===========================================================================
#   train_decomposition.py --------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import config.settings
import utils.format

import logging

import tensorflow as tf
import numpy as np
import os, sys
import matplotlib.pyplot as plt

#   settings ----------------------------------------------------------------
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"]="0"

logger = logging.getLogger("task:train-decomposition")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():
    # print user defined settings
    logger.debug("Print user defined settings")
    utils.format.print_data(config.settings._SETTINGS)
