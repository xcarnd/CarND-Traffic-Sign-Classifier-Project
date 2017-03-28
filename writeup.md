#**Traffic Sign Recognition** 

##Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./figures/dataset_stats.png "Number of data for each class."
[image2]: ./figures/dataset_visualization.png "Data set images visualization."
[image3]: ./figures/original_train.png "Original training input."
[image4]: ./figures/preprocessed_train.png "Preprocessed training input."
[image5]: ./figures/lnn.png "Curves for LNN."
[image6]: ./figures/2-layer-dnn.png "Curves for 2-Layer DNN."
[image7]: ./figures/lenet-5.png "Curves for original LeNet-5."
[image8]: ./figures/lenet-5-modified.png "Curves modified LeNet-5."
[image9]: ./figures/softmax_visulization.png "Softmax visualization"
[imaget1]: ./testing/s1.jpg "Test image 1"
[imaget2]: ./testing/s2.jpg "Test image 2"
[imaget3]: ./testing/s3.jpg "Test image 3"
[imaget4]: ./testing/s4.jpg "Test image 4"
[imaget5]: ./testing/s5.jpg "Test image 5"
[imaget6]: ./testing/s6.jpg "Test image 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

My solution for project 2 can be found [here](https://github.com/xcarnd/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second cell of the Jupyter notebook. I used the numpy library the get the summary statistics of the data set:

* The size of traininig set is *34799*.

* The size of test set is *4410*.

* The shape of a traffic sign image is *32 x 32*.

* The number of unique classes/labels in the data set is *43*.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the notebook. I plotted two figures. One for the number of data for each class, plotted as a bar chart; the other was a visualization for image data for each class.

![Dataset statistics][image1]

![Visualization][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 7th code cell of the notebook.

The main preprocessing routines are:

1. Convert the images to grayscale. I decided to do this because the actually meaning for a traffic sign is highly dependent on its content. Colors, on the other hand, are usually just for auxiliary(red usually for caution, blue usually for direction, etc).

2. Normalizing the content of the image. I took this step to avoid numerical issues during training.

Here is an example of an original image and preprocessed image:

![Original image][image3]

![Preprocessed image][image4]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)
	
I shuffled the training input before it was used in training. The validation and testing input data are used as they are provided(after preprocessing). 

All of the output data were transformed using One-Hot Encoding.

The code for the above is contained in the 8th code cell.

The final training set had 34799 images. The validation set has 4410 images. The test set has 12630 images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model can be found in the 11th -- 12th cell.

My final model consisted of the following layers:

| Layer             | Description                                     |
|:-----------------:|:-----------------------------------------------:|
| Input             | 32x32x1 Greyscale image                         |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x16     |
| Max polling       | 2x2 stride, outputs 14x14x16                    |
| RELU              |                                                 |
| Dropout           |                                                 |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 10x10x32     |
| Max polling       | 2x2 stride, outputs 5x5x32                      |
| RELU              |                                                 |
| Dropout           |                                                 |
| Flatten           | Flatten 5x5x32 to 800 neurons                   |
| Fully connected   | 800 neurons to 400 neurons                      |
| Fully connected   | 400 neurons to 168 neurons                      |
| Fully connected   | 168 neurons to 43 neurons                       |
| Softmax           |                                                 |

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 14th cell.

To train the model, I used:

1. A gradient descent optimizer to minimizing the loss of the network.
2. 50 samples per batch.
3. Train for 60 epochs.
4. 0.01 as the learning rate
5. Initialize the weights to be of mean 0 and standard deviation 0.1
6. 0.5 as the keep prob for both dropout layer when training.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I took the advices from my mentor and start working with this project using the simplest linear model. The linear model can get a pretty high (>90%) accuracy on the training set but low accuracy on the validation set (>70%), which indicates overfitting.

![Curves for LNN][image5]

The second model I used was a two-layer deep neural network, connected by RELU. Without dropouts, it performed worse than the simple linear model -- it overfit badly.

By adding a dropout layer, the model stopped suffering from the problem of overfitting. The accuracies were not too bad, yet not high enough (final train accuracy around 86%, valid accuracy around 85%), indicating, in my opinion, underfitting slightly.

![Curves for 2-Layer DNN][image6]

I've also tried adding one more layer. The final accuracies increased a little bit, not a significant improvement.

Finally I turned for my last model. It was based on the LeNet-5 architecture. Since LeNet-5 performed well on the MNIST datset(I learnt about this during the DNN tensorflow lab.) and traffic sign recognition is kind of like handwritten number recognition. In addition, unlike the deep neural networks, convolutional neural network can retain the spatial information of input image. So I thought it was more suitable for image recognition task.

I started with the original LeNet-5 architechture, with only the final output neurons changed to 43 to match the number of output classes. The loss of training set reduced quickly while training, but it overfit quickly too. High accuracy on train set and low accuracy on valid set made early-stopping not work.

![Original LeNet-5][image7]

So I changed the architechture a little bit:
1. Added dropouts to the fully connected layers to avoid overfitting.
2. Considering there were over 3 times more classes, I changed the layers of output for the convolutional layers to be 16/32, in the hope of giving more freedom to the network to discover more interesting features. The number of neurons were changed correspondingly.

And this was the final result:

![Final result][image8]

Accuracy on train set: 0.969453

Accuracy on valid set: 0.949887

Accuracy on test set: 0.940143


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

And just for fun, I've manually changed the first image a little bit and added as the 6th input. It looked like a speed limit of 20km/h instead of 70km/h. It was not a hard job for a human to figure out it was not 20km/h, and I wondered whether it could cheat the network.

![Speed limit-70][imaget1] ![Go straight or left][imaget2] ![Stop][imaget3] 
![Right-of-way at the next intersection][imaget4] ![Priority road][imaget5] ![Speed limit-70-m][imaget6] 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 17th cell of the notebook.

Here are the results of the prediction:

| Image                         | Prediction                                    | 
|:-----------------------------:|:---------------------------------------------:| 
| Speed limit(70km/h)           | Speed limit(70km/h)                           | 
| Go straight or left           | Go straight or left                           |
| Stop                          | Stop                                          |
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Priority road                 | Priority road                                 |
| Speed limit(70km/h)(modified) | Speed limit(70km/h)                           | 

The model could correctly predict all the traffic signs, giving an accuracy of 100%, even for the one I modified. While comparing with the accuracy on the test set which was, this was a reasonable result.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th and 19th cells of the notebook.

For most of the images, the model was pretty sure about its predictions. Probability of its output were all near to 1.

For the first image, the top five soft max were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.00000000e+00         | Speed limit (70km/h)                          |
|2.76116359e-14         | Speed limit (30km/h)                          |
|1.65766465e-14         | Speed limit (20km/h)                          |
|4.15310635e-19         | Speed limit (120km/h)                         |
|1.07647641e-21         | Stop                                          |

For the second image, the top five soft max were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|9.99999881e-01         | Go straight or left                           |
|6.10004278e-08         | No entry                                      |
|2.41754017e-09         | Roundabout mandatory                          |
|2.11643680e-09         | Speed limit (30km/h)                          |
|1.52307245e-09         | Stop                                          |
		  
For the third image, the top five soft max were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|9.98120129e-01         | Stop                                          |
|5.50784636e-04         | Traffic signals                               |
|4.93568834e-04         | Bumpy road                                    |
|2.60874367e-04         | Go straight or right                          |
|2.25307871e-04         | General caution                               |
		  
For the fourth image, the top five soft max were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|9.82789755e-01         | Right-of-way at the next intersection         |
|1.71781201e-02         | Priority road                                 |
|2.92915702e-05         | Roundabout mandatory                          |
|2.84285488e-06         | Beware of ice/snow                            |
|4.70099026e-09         | Double curve                                  |

For the fifth image, the top five soft max were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.00000000e+00         | Priority road                                 |
|1.10974847e-08         | Roundabout mandatory                          |
|6.38936473e-12         | End of all speed and passing limits           |
|8.95093678e-13         | Right-of-way at the next intersection         |
|1.66033284e-13         | End of no passing                             |

For the sixth image, the top five soft max were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.00000000e+00         | Speed limit (70km/h)                          |
|1.50303811e-10         | Speed limit (20km/h)                          |
|9.35809590e-12         | Speed limit (30km/h)                          |
|4.77220248e-17         | Speed limit (120km/h)                         |
|2.82361773e-19         | Stop                                          |

Softmaxes visualization are shown below. They're bar charts for softmax values for the input images, from top to bottom corresponding to test image 1 to test image 6:

![Softmax][image9]
