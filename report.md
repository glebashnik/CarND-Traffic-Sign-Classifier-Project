# Traffic Sign Recognition
---
## Overview

**Goal**: Build and evaluate a traffic sign recognition model.

The project consists of the following parts:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[dataset]: ./dataset.png "Visualization"
[sign1]: ./images/3.png "Traffic sign 1"
[sign2]: ./images/11.png "Traffic sign 2"
[sign3]: ./images/13.png "Traffic sign 3"
[sign4]: ./images/17.png "Traffic sign 4"
[sign5]: ./images/38.png "Traffic sign 5"
[conv1]: ./conv1.png "Traffic sign 4"
[conv2]: ./conv2.png "Traffic sign 5"

## Dataset exploration
### Data summary
I used *numpy* to generate a basic summary of the dataset: 
* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32)
* Number of classes = 43

### Exploratory Visualization
For visualization, I used *matplotlib* to generate a bar chart that shows a distribution of images in each class for training, validation and test sets. It is combined with visualization of a random images form each class and their labels.

![alt text][dataset]

## Design and Test a Model Architecture

### Preprocessing

I did normalizaiton of the data to scale color values between -1 and 1. 

After reading a relevant [research paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), where better performance is achieved on the grayscaled images, I also tried converting images to grayscal. However, there were no significant differences in the validation accuracy, so I remved greyscaling from the proecprocessing. Besides it makes sense that color might help to distinguish between signs, especially blue and red ones.

A possible improvement would be to convert images form RGB to HSV color space, where color similarities are captured more explicitly. Augumenting the dataset could also help. Noise to similuate image artifacts form the camer; rotation and translation to deal with broken or partially covered signs.

Another idea would be to apply 3d transformations, by projecting the image on 3d cube and rotating it along different axis. This would help to deal with images taken form different angles.

### Model Architecture

My final model consisted of the following layers:

| Layer         	| Description	        					            | 
|:-----------------:|:-----------------------------------------------------:| 
| Input         	| 32x32x3 RGB image   							        | 
| Convolution       | 5x5 size, 1x1 stride, valid padding, outputs 28x28x32 |
| RELU			    |												        |
| Max pooling	    | 2x2 stride, outputs 14x14x32 				            |
| Convolution       | 5x5 size, 1x1 stride, valid padding, outputs 10x10x64 |
| RELU			    |												        |
| Max pooling	    | 2x2 stride, outputs 5x5x64 	                        |
| Fully connected	| 256 size        									    |
| RELU				|        									            |
| Dropout			| 0.4 keep probability for training						|
| Fully connected	| 128 size        									    |
| RELU				|        									            |
| Dropout			| 0.4 keep probability for training						|
| Output        	| 43 size  												|


### Model Training

I used Adam optimizer, which is considered to be one of the best optimizers avaliable. During training, hyper-parameters were set to the following values.

* numer of epochs = 20
* batch size = 128
* rate = 0.001
* keep probability for training = 0.4

### Solution Approach

I started with the model and hyper-parameters from the [MNIST classificaiton with LeNet](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb)

Sign classification is a more complex task than classification of numbers in MNIST. The number of classes is increased from 10 to 43 and the images are captured in a wider variety of conditions. Therefore the first modificaiton I made is increased the number of feature maps in convolution layers and the size of fully connected layers.

For each epoch, I printed training and validation accuracy, noticing that training accuracy is close to 100% while valiadation accuracy is lagging behind, which might indicate overfitting. To make the mode more robuse, I added dropout after each fully connected layer. This decreased the gap between training and validation accuracy, achieving validation accuracy close to 97%.

After reading the research paper by [Sermanet and LeCun 2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that deals with the same task, I experimented with skip-layer connections. In this model, the output of the first and second convolution layers are combined as the input to the first fully connected layer. This change didn't effect my validation accuracy in a significant way so I rolled back to the previous model.

I did several iterations of trying out different values for hyper-parameters, decreasing the learning rate and increasing the number of epochs, increasing and decreasing the batch sizes. None of these tweaks yeilded improvements in the validation accuracy, except reducing the keep probability for the dropout to 0.4, which makes the model more robust.

**The final test accuracy achieved is 95.8%.**

## Test a model on new images

### Acquiring New images

To acquire new traffic sign images I used Google Street View, taking screenshots of traffic signs in Berlin. Here are the five images I got:

![sign1] ![sign2] ![sign3] ![sign4] ![sign5]

The images have similar quality to the images in the dataset so I would not expect any difficulties in classifying them. Image 4, the stop sign, can be a bit more difficult since it was captured form an angle.

### Performance on new images

All new images were classified correctly, achiving the **accuracy of 100%**.

### Model certainty

The model was 100% certain about all the images except for the stop sign image where the softmax probability was 99.6%, probably due to the angle as mentioned before.

## Visualizing the neural network

I visualized feature maps for the first and second convolution layers:

![conv1]
![conv2]

From the first convolution layer it is possible to see that some of the feature maps focus on foreground (feature map 19) vs background (feature map 2); edges (feature map 11) vs gradients (feature map 4). Interpretation of the feature maps from the second convolution layer is less clear.

