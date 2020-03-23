# ===========================================================================
#   patches.py --------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.tools.imgtools

import numpy as np
import tifffile

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class Patches():

    def __init__(self, img, obj="classification", categories=0, dtype=None, limit=None, margin=None, pad = None):
        self._patch = []
        self._patch_out = []
        self._img = img

        self._obj = obj
        if not dtype:
            if self._obj == "regression":
                dtype = np.float32
                self._categories = 0
            if self._obj == "classification":
                dtype = np.int16
                self._categories=range(categories)
                

        self._img_out = np.zeros((img.shape[0], img.shape[1]), dtype=dtype)
        
        self._limit = limit
        self._margin = margin
        self._pad = pad

        self.set_patch_limits()

    def __len__(self):
        return self._len

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self):
        if self._index < self._len-1:
            self._index += 1
            return self
        else:
            raise StopIteration

    def print_iter(self):
        print("Patch {} of {} patches...".format(self._index+1, self._len))

    @property
    def img(self):
        return self._img_out

    def get_image_patch_out(self, img):
        pad = self._c_pad
        img = img[0, pad[0][0]:-pad[0][1], pad[1][0]:-pad[1][1], :]
        img = dl_multi.tools.imgtools.expand_image_dim(img)

        patch = self._patch[self._index]
        patch_out  = self._patch_out [self._index]
        shape = img.shape
        return img[
            patch_out [0] - patch[0] : shape[0] + patch_out [1] - patch[1],
            patch_out [2] - patch[2] : shape[1] + patch_out [3] - patch[3],
            self._categories
        ] 

    def get_image_patch_out_max(self, img, axis=2):
        return np.argmax(self.get_image_patch_out(img), axis=axis)

    def get_image_patch(self, pad=None):
        patch = self._patch[self._index]
        img_patch = self._img[patch[0]:patch[1], patch[2]:patch[3]]

        pad = pad if pad else self._pad
        if pad:
            # pad to size divideble by 32
            self._c_pad =  self.get_image_pad(img_patch.shape, pad=pad)
            img_patch = np.pad(img_patch, (*self._c_pad, (0,0)), 'constant')
        return img_patch

    def get_image_pad(self, shape, pad=None):
        pad = pad if pad else self._pad
        pad_v = [int(pad/2) - int(shape[0] % pad / 2. + 0.5), int(pad/2) - int(shape[0] % pad / 2.)]
        pad_h = [int(pad/2) - int(shape[1] % pad / 2. + 0.5), int(pad/2) - int(shape[1] % pad  / 2.)]
        return (pad_v, pad_h)

    def get_sub_patch(self, shape, limit, margin):
        sub_patch = 1
        while (sub_patch * limit - (sub_patch-1)*margin) < shape : sub_patch = sub_patch + 1
        return sub_patch

    def set_patch(self, model_patch):
        patch_out = self._patch_out[self._index]

        if self._obj == "regression":
            patch = self.get_image_patch_out(model_patch)
        if self._obj == "classification":
            patch = self.get_image_patch_out_max(model_patch)

        patch = dl_multi.tools.imgtools.expand_image_dim(patch)

        self._img_out[patch_out[0]:patch_out[1], patch_out[2]:patch_out[3]] = patch[...,0] 
        
    def set_patch_limits(self, limit=None, margin=None):    
        limit = limit if limit else self._limit
        margin = margin if margin else self._margin
        sub_patch = [self.get_sub_patch(self._img.shape[0], limit[0], margin[0]), self.get_sub_patch(self._img.shape[1], limit[1], margin[1])]
        self._len = sub_patch[0] * sub_patch[1]

        px_max = 0
        overlapx = [0]
        stepsx = self._img.shape[1]/float(sub_patch[1])
        for px in range(sub_patch[1]):
            px_min = int(stepsx/2 + px*stepsx) - int(limit[0]/2)
            if px_min < 0: px_min=0      
            if px_max > 0: overlapx.append(px_max - px_min)    
            px_max = int(stepsx/2 + px*stepsx) + int(limit[0]/2)
            if px_max > self._img.shape[1]: px_max = self._img.shape[1]
            
            py_max = 0
            overlapy = [0]
            stepsy = self._img.shape[0]/float(sub_patch[0])
            for py in range(sub_patch[0]):
                py_min = int(stepsy/2 + py*stepsy) - int(limit[1]/2)
                if py_min < 0: py_min=0
                if py_max > 0: overlapy.append(py_max - py_min)
                py_max = int(stepsy/2 + py*stepsy) + int(limit[1]/2)
                if py_max > self._img.shape[0]: py_max = self._img.shape[0]
                
                self._patch.append([py_min, py_max, px_min, px_max])

        overlapx.append(0)
        overlapy.append(0)

        for px in range(sub_patch[1]):
            px_min = int(stepsx/2 + px*stepsx) - int(limit[0]/2) + int(overlapx[px]/2)
            if px_min < 0: px_min=0
            px_max = int(stepsx/2 + px*stepsx) + int(limit[0]/2) - int(overlapx[px+1]/2 + 0.5)
            if px_max > self._img.shape[1]: px_max = self._img.shape[1]
            
            for py in range(sub_patch[0]):
                py_min = int(stepsy/2 + py*stepsy) - int(limit[1]/2) + int(overlapy[py]/2)
                if py_min < 0: py_min=0
                py_max = int(stepsy/2 + py*stepsy) + int(limit[1]/2) - int(overlapy[py+1]/2 + 0.5)
                if py_max > self._img.shape[0]: py_max = self._img.shape[0]
                
                self._patch_out .append([py_min, py_max, px_min, px_max])
    
    def imsave(self, path):
        if self._obj == "classification":
            tifffile.imsave(path, self.img)
        if self._obj == "regression":
            tifffile.imsave(path, 
                #dl_multi.tools.imgtools.project_data_to_img(self.img)
                self.img
            )

    # def imsave_diff(self, path, truth):
    #     if self._obj == "classification":
    #         tifffile.imsave(path, self.img)
    #     if self._obj == "regression":
    #         tifffile.imsave(path,
    #             dl_multi.tools.imgtools.project_data_to_img( 
    #                 np.absolute(
    #                     dl_multi.tools.imgtools.project_data_to_img(self.img) -
    #                     dl_multi.tools.imgtools.project_data_to_img(truth)
    #                 )**2
    #             )
    #             # dl_multi.tools.imgtools.project_data_to_img( 
    #             #     np.absolute(
    #             #         self.img-truth
    #             #     )**2
    #             # )
    #         )