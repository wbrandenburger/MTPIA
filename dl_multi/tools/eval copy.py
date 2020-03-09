from PIL import Image

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ["CUDA_VISIBLE_DEVICES"]='3'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

import tensorflow as tf
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy.misc as sp
import scipy.ndimage as ndimage
#import cv2
import tifffile as tiff
from dl_multi.tools.tiramisu56 import tiramisu56  
  
def blubb():
  # print("huhu")
  ### old
  #out_cmap_arr = np.array([[255,255,255,255], # 0 Impervious surfaces
  #                      [0,0,255,255], # 1 Building
  #                      [0,255,0,255], # 2 Low vegetation
  #                      [0,255,255,255], # 3 Tree
  #                      [255,0,0,255], # 4 Car
  #                      [255,255,0,255], # 5 Clutter/background
  #                      ])/255.
                        
  # out_cmap_arr = np.array([[255,255,255,255], # 0 Impervious surfaces
  #                       [0,0,255,255], # 1 Building
  #                       [0,255,255,255], # 2 Low vegetation
  #                       [0,255,0,255], # Tree
  #                       [255,255,0,255], # 4 Car
  #                       [255,0,0,255], # 5 Clutter/background
  #                       [0,0,0,255] #6 boundary
  #                       ])/255.
  out_cmap_arr = np.array([[255,255,255,255], # 0 Impervious surfaces
                        [0,0,255,255], # 1 Building
                        [0,255,255,255], # 2 Low vegetation
                        [0,255,0,255], # Tree
                        [255,255,0,255], # 4 Car
                        [255,0,0,255], # 5 Clutter/background
                        [0,0,0,255] #6 boundary
                        ])/255.
  out_cmap = clr.ListedColormap(out_cmap_arr)

  #test_data = '/media/Raid/matthias/tensorflow/PIA2019/eval/vaihingen_test.txt'

  img_dir = '/media/Raid/Datasets/Vaihingen/top/'
  dsm_dir = '/media/Raid/Datasets/Vaihingen/dsm/'
  lbl_dir = '/media/Raid/Datasets/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/'
  #lbl_dir = '/media/Raid/Datasets/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE/'


  # model_dir = '/media/Raid/matthias/tensorflow/PIA2019/checkpoints/PIA/'
  model_dir = "B:\\DLMulti\\checkpoints\\train-2epch"
  for name in range(1): #os.listdir(model_dir):
    
    model_name = model_dir + '\\pia.ckpt-10'
    # if not '_FOR_' in model_name: continue
    # print(name)
    
    writedir = "B:\\DLMulti\\eval\\" 
    if not os.path.isdir(writedir): os.mkdir(writedir)
    
    logfile = open("B:\\DLMulti\\logs\\train-2epch\\log.txt", "w+")
    
    # get test examples
    #lines = [line.rstrip('\n') for line in open(test_data)]
    
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
      
      image_orig = np.array(Image.open("A:\\Datasets\\Vaihingen\\attempt\\top_mosaic_09cm_area1_s.tif" )).astype(np.float32)
      dsm = np.array(tiff.imread("A:\\Datasets\\Vaihingen\\attempt\\dsm_09cm_matching_area1_s.tif"))
      label = np.array(Image.open("A:\\Datasets\\Vaihingen\\attempt\\truth_mosaic_09cm_area1_s.tif"))

      annotation = np.zeros((label.shape[0], label.shape[1]))
      annotation = np.all(label == [0,0,255], axis=-1) + 2*np.all(label == [0,255,255], axis=-1) + 3*np.all(label == [0,255,0], axis=-1) + 4*np.all(label == [255,255,0], axis=-1) + 5*np.all(label == [255,0,0], axis=-1)
      ## with borders
      #annotation = np.all(label == [0,0,255], axis=-1) + 2*np.all(label == [0,255,255], axis=-1) + 3*np.all(label == [0,255,0], axis=-1) + 4*np.all(label == [255,255,0], axis=-1) + 5*np.all(label == [255,0,0], axis=-1) + 6*np.all(label == [0,0,0], axis=-1)

      patchi_x = 256
      patchi_y = 256
      patchi_x_i = 32
      patchi_y_i = 32
      ## compute limits for patches
      patch_limits = []
      amax_limits = []
      # print(image_orig.shape)
      p_in_x = 1
      while (p_in_x*patchi_x - (p_in_x-1)*patchi_x_i) < image_orig.shape[1]: p_in_x = p_in_x + 1
      p_in_y = 1
      while (p_in_y*patchi_y - (p_in_y-1)*patchi_y_i) < image_orig.shape[0]: p_in_y = p_in_y + 1
      print(p_in_x)
      print(p_in_y)
      px_max = 0
      overlapx = [0]
      stepsx = image_orig.shape[1]/float(p_in_x)
      for px in range(p_in_x):
        px_min = int(stepsx/2 + px*stepsx) - int(patchi_x/2)
        if px_min < 0: px_min=0      
        if px_max > 0: overlapx.append(px_max - px_min)
        
        px_max = int(stepsx/2 + px*stepsx) + int(patchi_x/2)
        if px_max > image_orig.shape[1]: px_max=image_orig.shape[1]
        
        py_max = 0
        overlapy = [0]
        stepsy = image_orig.shape[0]/float(p_in_y)
        for py in range(p_in_y):
          py_min = int(stepsy/2 + py*stepsy) - int(patchi_y/2)
          if py_min < 0: py_min=0
          if py_max > 0: overlapy.append(py_max - py_min)
          py_max = int(stepsy/2 + py*stepsy) + int(patchi_y/2)
          if py_max > image_orig.shape[0]: py_max=image_orig.shape[0]
        
          patch_limits.append([py_min, py_max, px_min, px_max])
          #print([px_min, px_max, py_min, py_max])    count = count + 1


      overlapx.append(0)
      overlapy.append(0)
    
      for px in range(p_in_x):
        px_min = int(stepsx/2 + px*stepsx) - int(patchi_x/2) + int(overlapx[px]/2)
        if px_min < 0: px_min=0
        px_max = int(stepsx/2 + px*stepsx) + int(patchi_x/2) - int(overlapx[px+1]/2 + 0.5)
        if px_max > image_orig.shape[1]: px_max=image_orig.shape[1]
        
        for py in range(p_in_y):
          py_min = int(stepsy/2 + py*stepsy) - int(patchi_y/2) + int(overlapy[py]/2)
          if py_min < 0: py_min=0
          py_max = int(stepsy/2 + py*stepsy) + int(patchi_y/2) - int(overlapy[py+1]/2 + 0.5)
          if py_max > image_orig.shape[0]: py_max=image_orig.shape[0]
        
          amax_limits.append([py_min, py_max, px_min, px_max])
      
      # print(overlapx, overlapy)
      # print(patch_limits)
      # print(amax_limits)
      
      #for blub in range(len(patch_limits)):
      #  print([amax_limits[blub][0] - patch_limits[blub][0],
      #          amax_limits[blub][1] - patch_limits[blub][1],
      #          amax_limits[blub][2] - patch_limits[blub][2],
      #          amax_limits[blub][3] - patch_limits[blub][3]])
      
      amax = np.zeros((image_orig.shape[0], image_orig.shape[1]), np.int16)
      
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
          pred = tiramisu56(data)
        
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        
        with tf.Session() as sess:

          sess.run(init_op)
          saver.restore(sess, model_name)
          sess.graph.finalize()
          
          ## orig
          all_out = sess.run([pred])
          # print(all_out)
          # print(pad_v)
          # print(pad_h)
          # print(len(all_out[0]))
          # print(all_out[0][0].shape)
          out = all_out[0][0][
            0,pad_v[0]:-pad_v[1],
            pad_h[0]:-pad_h[1],:
          ]
        
          #resized_out = np.zeros((label.shape[0],label.shape[1],7))
          
          cur = np.argmax(out[amax_limits[p][0] - patch_limits[p][0]:out.shape[0]+amax_limits[p][1] - patch_limits[p][1], 
                              amax_limits[p][2] - patch_limits[p][2]:out.shape[1]+amax_limits[p][3] - patch_limits[p][3],
                              :], axis=2)
          
          
          amax[ amax_limits[p][0]:amax_limits[p][1], amax_limits[p][2]:amax_limits[p][3]] = cur
        
      ## optional: copy borders
      #amax = np.where(annotation==6, 6, amax)
      
      plt.imshow(image_orig.astype(np.uint8))
      plt.figure()
      plt.imshow(amax, cmap=out_cmap, clim=(0,6))
      plt.show()
      
      writename = "B:\\DLMulti\\eval\\result.png"
      plt.imsave(writename, amax, cmap=out_cmap, vmin=0, vmax=6)

      #####
      #plt.imsave("/media/klein/Paper/2019_GSW/results_ft1_bs8/"+l[:-3]+"png", amax, cmap=out_cmap)
      #####
      true = np.count_nonzero( amax==annotation)
      # ignore pixel with value 6
      acc = (true - np.sum(amax==6))/(annotation.shape[0]*annotation.shape[1] - np.sum(annotation==6))
      
      for c in range(6):
        current_out = amax==c
        current_gt = annotation==c
        
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
