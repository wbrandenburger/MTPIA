# ===========================================================================
#   imgio.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.__init__ import _logger
import dl_multi.tools.imgtools
import dl_multi.utils.general
import dl_multi.tools.imgcontainer

import numpy as np
import PIL
import shutil
import tifffile

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def read_image(
        path
    ):

    _logger.debug("[READ] '{}'".format(path))
    
    if str(path).endswith(".tif"):
        img = tifffile.imread(path)
    else:
        img = np.asarray(PIL.Image.open(path))
    
    return img
    
#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def save_image(
        dest,  
        img
    ):
    
    _logger.debug("[SAVE] '{}'".format(dest))

    if str(dest).endswith(".tif"):
        tifffile.imwrite(dest, img)
    else:
        PIL.Image.fromarray(img).write(dest)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def copy_image(
        path,  
        dest
    ):
    
    dl_multi.__init__._logger.debug("[COPY] '{}'".format(dest))
    shutil.copy2(path, dest)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_image(
        path, 
        spec,
        param_label=dict(), 
        scale=100, 
        show=False, 
        **kwargs
    ):

    img = dl_multi.tools.imgtools.resize_img(read_image(path), scale)

    if param_label and spec == "label":
        img = dl_multi.tools.imgtools.labels_to_image(img, param_label)

    if show:
        img = dl_multi.tools.imgtools.project_data_to_img(img, dtype=np.uint8, factor=255)
    if show:
        img =  dl_multi.tools.imgtools.stack_image_dim(img)

    return img

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_data(
        files,
        specs, # list()
        param_io,
        param_label=dict(), 
        param_show=dict(), # scale=100, show=False, live=True, 
        param_log=dict() # log_dir, ext=".log"
        # default_spec="image",       
    ):

    load = lambda path, spec: dl_multi.tools.imgio.get_image(
        path, 
        spec, 
        param_label=param_label, 
        **param_show # scale=100, show=False, live=True
    )
    
    img_in = list() 
    for f_set in files:
        img = dl_multi.tools.imgcontainer.ImgListContainer(
            load=load, 
            log_dir=param_log["path_dir"]# log_dir, ext=".log"
        )
        # if specs:
        #     for f in f_set:
        #     img.append(path = f, spec=default_spec)      
        # else:
        for f, s in zip(f_set, specs):
            img.append(path = f, spec=s)

        img_in.append(img)

    get_img_path = dl_multi.utils.general.PathCreator(**param_io)
    img_out = lambda path, img, **kwargs: save_image(get_img_path(path, **kwargs), img)

    log_out = lambda path, **kwargs : get_img_path(path, **param_log, **kwargs)

    return img_in, img_out, log_out
