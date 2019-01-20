# Training Convolutional Neural Networks and Compressed Sensing End-to-End for Microscopy Cell Detection

<img src="https://github.com/yaoxuexa/End2End/blob/master/sample.jpg" width = "800" height = "500" div align=center>

**Introduction**

Automated cell detection and localization from microscopy images are significant tasks in biomedical research and clinical practice. In this paper, we design a new cell detection and localization algorithm that combines deep convolutional neural network (CNN) and compressed sensing (CS) or sparse coding (SC) for end-to-end training. The key innovation behind our algorithm is that the cell detection task is structured as a point object detection task in computer vision, where the cell centers (i.e., point objects) occupy only a tiny fraction of the total number of pixels in an image. Thus, we can apply compressed sensing (or, equivalently sparse coding) to compactly represent a variable number of cells in a projected space. Subsequently, CNN regresses this compressed vector from the input microscopy image. The SC/CS recovery algorithm (L1 optimization) can then recover sparse cell locations from the output of CNN. We train this entire processing pipeline end-to-end and demonstrate that end-to-end training improves accuracy over a training paradigm that treats CNN and CS-recovery layers separately.

**Prerequisites**

Download the [data](http://tupac.tue-image.nl/node/3) from MICCAI grand challenge of mitosis detection.

Download and install the python package [scikit-image](https://scikit-image.org/) for radon transform and inverse radon transform.

Our implementation is using the deep learning library: Tensorflow. See the instructions [here](https://www.tensorflow.org/install/) for a step-by-step guide for installation on Ubuntu.

**Usage**

AMIDA dataset comes from a mitosis detection challenge held by MICCAI at 2016. It contains 587 whole-slide training images and 34 whole-slide testing images in size of 2000*2000 pixels. Images and ground-truth cell centroid coordinates of training set are in mitoses_image_data_part_1.zip, mitoses_image_data_part_2.zip, mitoses_image_data_part_3.zip and mitoses_ground_truth.zip. Testing images are in mitoses-test-image-data.zip. Unzip them into the "data/original/" folder.

So far, ground-truth cell centroid coordinates of the testing set is not released by the contest organizer. To get our model's detection results on the testing set of AMIDA-2016 dataset. You can firstly unzip AMIDA16-ECNNCS.zip then run "ECNNCS_test.m" to visualize every detected mitosis by our trained model on all the 34 whole-slide testing images.

For training the model on your own data, the procedures are as follows:

Run ECNNCS_encode_trainset.m to generate your training image list and encode the ground-truth cell coordinates to label vectors. You can also perform data augmentation using this .m file to supplement your training set, if necessary.

cd to the home of project, 

train the model by "python ECNNCS_train.py /path-to-trainList-file /path-to-validationList-file /path-of-model-saved-to"

contact: yxue2@ualberta.ca
