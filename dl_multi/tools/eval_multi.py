# ===========================================================================
#   train.py ----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.tools.imgtools

from PIL import Image

import os
import logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

# logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

import tensorflow as tf
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy.misc as sp
import scipy.ndimage as ndimage
#import cv2
import tifffile as tiff
import dl_multi.tools.tiramisu56

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def eval(param_eval, param_label, param_color, param_out): 
    
  dl_multi.__init__._logger.debug("Start training single task model with settings:\n'param_train':\t'{}',\n'param_out':\t'{}'".format(param_eval, param_out))

  #   settings ------------------------------------------------------------
  # -----------------------------------------------------------------------
  checkpoint = param_out["checkpoints"] + "\\" + param_eval["checkpoint"]
  output_label = param_out["eval"] + "-" + param_eval["checkpoint"] + "_CLS.png"
  output_height = param_out["eval"] + "-" + param_eval["checkpoint"] + "_HGHT.png"

  logfile = open(
    param_out["logs"]  + "\\" + param_eval["checkpoint"] + ".eval.log", "w+"
  )
  out_cmap = clr.ListedColormap(np.array(param_color)/255.)
  
  for name in range(1):

    model_name = checkpoint
  
    count = 0
    sum_acc = 0

    #[out, true, gt, iou]
    classes = np.zeros((6, 4))
    
    ## iterate all test examples
    for l in range(1):
      # print(l)
      logfile.write("%s\n" % l)
      count = count + 1
      
      # image_orig = np.array(Image.open(img_dir + l)).astype(np.float32)
      # dsm = np.array(tiff.imread(dsm_dir + "dsm_09cm_matching_" + l[16:]))
      # label = np.array(Image.open(lbl_dir + l))
      
      image_orig = np.array(Image.open("B:\\DLMulti\\eval\\top_mosaic_09cm_area1_s.tif" )).astype(np.float32)
      # dsm = np.array(tiff.imread(B:\DLMulti\attempt\\dsm_09cm_matching_area1_s.tif"))
      label = dl_multi.tools.imgtools.labels_to_image(
        np.array(Image.open("B:\\DLMulti\\eval\\truth_mosaic_09cm_area1_s.tif")), param_label) 

      patch_limits, amax_limits = dl_multi.tools.patches.get_patch_limits(
        image_orig.shape, 256, 256, 32, 32
      )
      amax = np.zeros((image_orig.shape[0], image_orig.shape[1]), np.int16)
      bmax = np.zeros((image_orig.shape[0], image_orig.shape[1]), np.int16)
      for p in range(len(patch_limits)):
      
        tf.reset_default_graph()
        tf.Graph().as_default()
              
        image=image_orig[patch_limits[p][0]:patch_limits[p][1], patch_limits[p][2]:patch_limits[p][3],:]
      
        # pad to size divideble by 32
        pad_h = [ 16-int(image.shape[1] % 32 / 2. + 0.5), 16-int(image.shape[1] % 32 / 2.)]
        pad_v = [ 16-int(image.shape[0] % 32 / 2. + 0.5), 16-int(image.shape[0] % 32 / 2.)]
        #image = np.pad(image, (pad_v, pad_h, (0,0)), 'constant')
        # print(pad_h, pad_v)
        lout = image
        image = np.pad(image, (pad_v, pad_h, (0,0)), 'constant')

        image = image[:,:,:] / 127.5 - 1.
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image,0)

        data = image
        
        #print(data.shape)
        with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
        ## orig
          pred = dl_multi.tools.tiramisu56.tiramisu56(data)
        
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        
        with tf.Session() as sess:

          sess.run(init_op)
          saver.restore(sess, checkpoint)
          sess.graph.finalize()
          
          ## orig
          all_out = sess.run([pred])
          a_out = all_out[0][0][
            0,pad_v[0]:-pad_v[1],
            pad_h[0]:-pad_h[1],:
          ]

          b_out = all_out[0][2][
            0,pad_v[0]:-pad_v[1],
            pad_h[0]:-pad_h[1],0
          ]
          #resized_out = np.zeros((label.shape[0],label.shape[1],7))
          
          a_cur = np.argmax(
            a_out[
              amax_limits[p][0] - patch_limits[p][0]:a_out.shape[0]+amax_limits[p][1] - patch_limits[p][1],
              amax_limits[p][2] - patch_limits[p][2]:a_out.shape[1]+amax_limits[p][3] - patch_limits[p][3],
              :
            ], 
            axis=2
          )

          b_cur = b_out[
            amax_limits[p][0] - patch_limits[p][0]:b_out.shape[0]+amax_limits[p][1] - patch_limits[p][1], 
            amax_limits[p][2] - patch_limits[p][2]:b_out.shape[1]+amax_limits[p][3] - patch_limits[p][3]
          ]
          
          
          amax[ amax_limits[p][0]:amax_limits[p][1], amax_limits[p][2]:amax_limits[p][3]] = a_cur
          bmax[ amax_limits[p][0]:amax_limits[p][1], amax_limits[p][2]:amax_limits[p][3]] = b_cur
        
      ## optional: copy borders
      #amax = np.where(annotation==6, 6, amax)
      
      plt.imsave(output_label, amax, cmap=out_cmap, vmin=0, vmax=6)
      plt.imsave(output_height, bmax)

      true = np.count_nonzero( max==label)
      # ignore pixel with value 6
      acc = (true - np.sum(amax==6))/(label.shape[0]*label.shape[1] - np.sum(label==6))
      
      for c in range(6):
        current_out = amax == c
        current_gt = label == c
        
        #plt.figure()
        #plt.imshow(current_out)
        #plt.figure()
        #plt.imshow(current_gt)
        #plt.figure()
        #plt.imshow(np.logical_and(current_out, current_gt))
        #plt.figure()
        #plt.imshow(current_out^current_gt)
        #plt.imshow(out[:,:,c])
        #plt.show()
        
        classes[c,0] = classes[c,0] + np.count_nonzero(current_gt)
        classes[c,1] = classes[c,1] + np.count_nonzero(current_out)
        classes[c,2] = classes[c,2] + np.count_nonzero(np.logical_and(current_out, current_gt))      
        classes[c,3] = classes[c,3] + np.count_nonzero(np.logical_or(current_out, current_gt))
              
      sum_acc = sum_acc + acc
      
      print(str(acc))    
      logfile.write("Acc: %2.5f\n" %acc )

    print(name) 
    print("Average (over img) Acc: " + str(sum_acc/count))
    
    logfile.write("\n\nAverage (over img) Acc: %2.5f\n" %(sum_acc/count) )

    overall = 0
    px = 0
    for c in range(6):
      overall = overall + classes[c,2]
      px = px + classes[c,0]

    print("Overall: " + str(overall/px))
    logfile.write("\n\Overall Acc: %2.5f\n" %(overall/px) )

    print("     gt      out     true     union")
    print(classes)
    print("Accuracy: " + str(classes[:,2]/classes[:,0]))
    print("IoU: " + str(classes[:,2]/classes[:,3]))
    
    logfile.close()