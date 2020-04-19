# ===========================================================================
#   normalization.py --------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import numpy as np

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def default(x):
    return x

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def add_value(x, value=0):
    return x + value

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def standardize_per_input(x, source_range=list(), dest_range=[-1,1], dtype=np.float32):
    x_normed = x.astype(dtype)
    
    if not source_range:
        source_range = [np.min(x_normed), np.max(x_normed)]

    x_normed = (x_normed-source_range[0]) / (source_range[1]-source_range[0])
    x_normed = x_normed * (dest_range[1]-dest_range[0]) + dest_range[0]

    return x_normed

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def normalize_per_input(x, epsilon=1e-8, dtype=np.float32):
    x_normed = x.astype(dtype)
    
    mean, variance = np.mean(x), np.var(x)
    x_normed = (x - mean) / np.sqrt(variance + epsilon) # epsilon to avoid dividing by zero

    return x_normed