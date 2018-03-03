# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/center.jpg "Center Cam"
[image2]: ./figures/left.jpg "Left Cam"
[image3]: ./figures/right.jpg "Right Cam"
[image4]: ./figures/center.flip.jpg "Center Cam - flip"
[image5]: ./figures/left.flip.jpg "Left Cam - flip"
[image6]: ./figures/right.flip.jpg "Right Cam - flip"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 the movement of the simulated car in autonomous mode.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

I tried 2 different model architectures, viz. the NVIDIA and the comma.ai architectures. I found that the NVIDIA architecture performed better. I also tried implementing the NVIDIA architecture using keras functional api, but I wasn't satisfied with the result. Therefore, I discuss here only the sequential nvivia architecture.

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 5x5 filter sizes and depths 34, 36 and 48 respectively, another and 2 3x3 filter sizes and depths of 64 each (model.py lines 111-116). 3 fully connected layers follows the convolution layers of sizes 100, 50 and 10 respectively (model.py lines 117-121). The output is then obtained at a single node after the fully connected layers (model.py line 122).

The model includes RELU (for convolution layers) and ELU (for the fully connected layers) to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer (model.py line 110). Also top 70 (the sky) and bottom 25 lines (car bonnet) are cropped out for simplifying the image input to the model (model.py line 109)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 118, 120). 

The model was trained and validated on real as well as augmented data sets to ensure that the model was not overfitting (code line 60-71). Augmenting was mostly done simply by using left and right camera images with an approximate correction steering value, and by flipping the images to have a fairly equal number of left and right turns. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 168).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to design-implement-test-modify a number of networks and check for their performance.

My first step was to simply use a simple convolution neural network model with a few fully connected layers and produce a single output. I thought the model would be appropriate as it is a regression problem predicting the steering angles based on the viewport frame of the track. This model worked well until the bridge where it bumped into the walls. Later I implemented the NVIDIA cnn architecture, where the model performed exceptionally well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I introduce dropouts in between the fully connected layers and it did the magic.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I retrained the model with a few more epochs. This solved the issue.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture summary is described in the following table. 

|Sequence  |Layer (type)                 |Output Shape              |Param #   |
|:---------|:----------------------------|:-------------------------|:---------|
|1         |cropping2d_17 (Cropping2D)   |(None, 65, 320, 3)        |0         |
|2         |lambda_19 (Lambda)           |(None, 65, 320, 3)        |0         |
|3         |conv2d_83 (Conv2D)           |(None, 31, 158, 24)       |1824      |
|4         |RELU                         |                          |          |
|5         |conv2d_84 (Conv2D)           |(None, 14, 77, 36)        |21636     |
|6         |RELU                         |                          |          |
|7         |conv2d_85 (Conv2D)           |(None, 5, 37, 48)         |43248     |
|8         |RELU                         |                          |          |
|9         |conv2d_86 (Conv2D)           |(None, 3, 35, 64)         |27712     |
|10        |RELU                         |                          |          |
|11        |conv2d_87 (Conv2D)           |(None, 1, 33, 64)         |36928     |
|12        |RELU                         |                          |          |
|13        |flatten_18 (Flatten)         |(None, 2112)              |0         |
|14        |dense_63 (Dense)             |(None, 100)               |211300    |
|15        |ELU                          |(None, 100)               |0         |
|16        |dense_64 (Dense)             |(None, 50)                |5050      |
|17        |ELU                          |(None, 50)                |0         |
|18        |dense_65 (Dense)             |(None, 10)                |510       |
|19        |dense_66 (Dense)             |(None, 1)                 |11        |

#### 3. Creation of the Training Set & Training Process

Since I was constrained by a limited hardware capabilities. I wasn't ab;t to capture data by my own. Instead, I augmented the available pre-recorded data by using the left and right camera images with some steering correction of +0.25 and -0.25 respectively. Further more I flipped the images and negated the corresponding steering angles, so finally I was left with 6 times the original datapoints in the dataset.

1. Center Image\
![image1]
2. Left Image\
![image2]
3. Right Image\
![image3]


1. Center Image - Flipped\
![image4]
2. Left Image - Flipped\
![image5]
3. Right Image - Flipped\
![image6]


After the collection process, I had 48216 number of data points. I then preprocessed this data to crop out the top 70 rows and bottom 25 rows of the image. Then I normalized the data between -1 and 1.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used 6 epochs to train my model. I used an adam optimizer so that manually training the learning rate wasn't necessary.
