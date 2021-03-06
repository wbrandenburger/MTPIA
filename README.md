# MTPIA

Implementation of a multi task environment for semantic segmentation and digital surface models.

## Settings

## Documentation

### Model Settings

```Python
batch_size_vaihingen=2 # batch size
image_train_size = [448, 448] # image size
num_epochs = 150000 ## in fact num_iterations but tensorflow needs (needed) num_epochs
```

- [Batch size](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
  - The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters
  - When all training samples are used to create one batch, the learning algorithm is called batch gradient descent. When the batch is the size of one sample, the learning algorithm is called stochastic gradient descent. When the batch size is more than one sample and less than the size of the training dataset, the learning algorithm is called mini-batch gradient descent.
    - **Batch Gradient Descent:** Batch Size = Size of Training Set
    - **Stochastic Gradient Descent:** Batch Size = 1
    - **Mini-Batch Gradient Descent:** 1 < Batch Size < Size of Training Set
  - [Configure batch size](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)
  - [Effects of batch size](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/)

### Folder Settings

```Python
name = 'PIA/vaihingen_FOR_dsm_190711_DLBox_wd0_09'
checkpoints_dir = '/media/Raid/matthias/tensorflow/PIA2019/checkpoints/'+name
log_folder = '/media/Raid/matthias/tensorflow/PIA2019/log/'+name

tfrecords_filename_vaihingen = '/media/Raid/matthias/tensorflow/PIA2019/vaihingen_w_dsm.tfrecords'
```

### Model process

- Get a queue with the output strings from tf-record and based on this decode trainings data (images, annotations and DSM)
- Create batches with trainings data and details of maximum and minimum elements in queue
- Define the [training model](source\tiramisu56_vaihingen_FOR_dsm.py)
- Define losses (cross entropy for categorical data and regression loss for interval data)
- Loop over defined number of epochs and saving checkpoints

### Frequently Asked Questions

- [tf-slim](https://github.com/google-research/tf-slim): TF-Slim is a lightweight library for defining, training and evaluating complex models in TensorFlow

### Improvements - Data IO

[Better improvement with tf-record](source\tf_records_vaihingen.py) for providing image data in an efficient format.

- [Create a tf-record filenames queue](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- While [train the model](source\train_vaihingen_FOR_dsm.py) get a queue with the output strings.
- [Decode](source\tf_records_vaihingen.py) images and annotations with resulting keys

### Improvements - Augmentation

- [Augmentation functions](source\augmentation.py)
  - Random cropping and flipping of images
  - Random cropping and flipping of DSM's
  - Random cropping, rotating and flipping of DSM's
  - Random cropping, rotating 90 degrees and flipping of DSM's
  - Random cropping, rotating and flipping of IR-images
  - Add color and noise
  - Random crops from original tiles (not implemented)

### Evaluation

```Python
test_data = '/media/Raid/matthias/tensorflow/PIA2019/eval/vaihingen_test.txt' # subset of images defining test data

img_dir = '/media/Raid/Datasets/Vaihingen/top/' # image folder
dsm_dir = '/media/Raid/Datasets/Vaihingen/dsm/' # DSM folder
lbl_dir = '/media/Raid/Datasets/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/' # ground truth folder
model_dir = '/media/Raid/matthias/tensorflow/PIA2019/checkpoints/PIA/' # checkpoints
```

In loop:

```Python
writedir = "/media/Raid/matthias/tensorflow/PIA2019/eval/results_train/"
```

## Problems of different resolutions

Training can be conducted via different strategies:

- upsampling images with lower resolution to the same resolution of the input image before performing crops,
- downsampling the images with higher resolution to speed up training and testing.

## Problems of cropping and tiling

Inference is implemented using a Gaussian prior over patches to avoid a checkerboard effect on the output. We predict patches sequentially with a stride smaller than the window size and weight overlapping areas with a 2D Gaussian map. Results are improved when using bigger windows and small strides as we can leverage more information from neighbor patches. For our experiments, we use a test window of 1024 and a stride of 256.
