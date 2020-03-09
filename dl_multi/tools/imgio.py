# ===========================================================================
#   imgio.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.tools.imgtools

import numpy as np
import PIL
import shutil
import tifffile

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def read_image(path):
    dl_multi.__init__._logger.debug("[READ] '{}'".format(path))
    
    if str(path).endswith(".tif"):
        return tifffile.imread(path)
    else:
        return np.asarray(PIL.Image.open(path))
    
#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def save_image(dest,  img):
    dl_multi.__init__._logger.debug("[SAVE] '{}'".format(dest))

    if str(dest).endswith(".tif"):
        tifffile.imwrite(dest, img)
    else:
        PIL.Image.fromarray(img).write(dest)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def copy_image(path,  dest):
    dl_multi.__init__._logger.debug("[COPY] '{}'".format(dest))
    shutil.copy2(path, dest)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_image(path, spec="image", param_label=dict(), scale=100, show=False):
    img = dl_multi.tools.imgtools.resize_img(read_image(path), scale)
    if param_label and spec == "label":
        img = dl_multi.tools.imgtools.labels_to_image(img, param_label)
    if show and spec in ["label", "height", "msi"]:
        img = dl_multi.tools.imgtools.project_data_to_img(img)
    if show:
        img =  dl_multi.tools.imgtools.stack_image_dim(img)
    return img
