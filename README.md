# Traffic Sign Recognition
---
[//]: # (Image References)

[image1]: ./examples/data_distribution.jpg "Class Distribution"
[image2]: ./examples/pre-processed.png "Preprocessed"
[image3]: ./examples/augmented.png "Augmented Images"
[image4]: ./examples/sign_1.png "Traffic Sign 1"
[image5]: ./examples/sign_2.png "Traffic Sign 2"
[image6]: ./examples/sign_3.png "Traffic Sign 3"
[image7]: ./examples/sign_4.png "Traffic Sign 4"
[image8]: ./examples/sign_5.png "Traffic Sign 5"
[image9]: ./examples/actual_test_images.png "Scaled Signs"

Here is a link to my [project code](https://github.com/brett--anderson/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualisation of the data set. It is a bar chart showing how many examples of each class there is in the training validation and test sets. The distributions show that there is an over-representation of some classes that would lead to a bias of classifying new examples into the higher represented classes. However, since the biased representation is almost exactly the same across the training, validation and test sets obtaining a good test accuracy is possible. This bias may be a problem when generalising to downloaded images obtained from the internet.

![distribution of classes in training, validation and test sets][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images from RGB to YUV and then just used the Y channel for my images. The YUV scale skews the colours to better approximate human perception of colours and the Y channel encodes the brightness / intensity of the colours. I did this as I had better results and I've found in other projects that trying to get the machine to see colours the way humans do really helps. This makes sense for road signs since they user colours like yellow which humans are easily able to differentiate.

I also normalised the image using the z-score: (value - mean) / standard deviation. I found that I got better results using this formula than when I used the min-max normalisation formula. Below are example of the images after pre-processing

![examples of pre-processed images][image2]

I decided to generate additional data using the imgaug library because it allowed the model to train with more robust data representing more real-world examples of skewed of degraded images of traffic signs. It also provided the quantity of data required to accurately train my deep model which has a decent number of layers and therefore parameters that need to determined. Ultimately my model architecture provided 95% accuracy without augmentation and 98% with augmentation. I also found that I need to normalise the images into the range [0-255] and cast to uint8 data types for the augmentation to work. Then I needed to return the images to the original z-score normalised floats to keep the values small and centred around the origin so that the optimiser could work efficiently.

Here are examples of the additional augmentations for a single image

![exhaustive set of augmented images based on a single origin image][image3]

I performed the following augmentations:

 1. crop images from each side by 0 to 6px
 2. blur images with a sigma between 0 and 2.0
 3. scale images to 90-110% of their size, individually per axis
 4. translate by -10 to +10 percent (per axis)
 5. rotate by -10 to +10 degrees
 6. shear by -10 to +10 degrees

I was careful not to completely flip the images during augmentation as a 'no right turn' could become a 'no left turn' and would then have the wrong label in the training data.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 YUV image with only 'Y' channel       | 
| Convolution 5x5       | 1x1 stride, same padding, outputs 32x32x32    |
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, same padding, outputs 32x32x32    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 16x16x32                 |
| Dropout               | 0.5 During training, 0 when test/val          |
| Convolution 5x5       | 1x1 stride, same padding, outputs 16x16x64    |
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, same padding, outputs 16x16x64    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 8x8x64                   |
| Dropout               | 0.7 During training, 0 when test/val          |
| Convolution 5x5       | 1x1 stride, same padding, outputs 8x8x128     |
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, same padding, outputs 8x8x128     |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 4x4x128                  |
| Dropout               | 0.65 During training, 0 when test/val         |
| Flatten               | flatten the output of the first double conv   |
| Flatten               | flatten the output of the second double conv  |
| Flatten               | flatten the output of the third double conv   |
| Merge                 | Merge the three flattened output into one     |
| Fully Connected       | In 14336 Out 1024                             |
| RELU                  |                                               |
| Fully Connected       | In 1024 Out 1024                              |
| RELU                  |                                               |
| Fully Connected       | In 1024 Out num classes (43)                  |
| Softmax               |                                               |

I included back to back convulation layers as I read that they are able to handle more non-linear patterns in the data. I also found that they improved my accuracy. I also borrowed some of the structure from the resnet and inception architectures where earlier convolution blocks can ouput to the final fully connected layers, as well as the next convolution block. This should allow the fully connected layers to examine some of the local structures in the images, as well as the final global assortments of those structures.
 
I also included three drop out layers to force the network to be robust and not rely on an individual signal to make a decision. I found I had the best results when I lowered the drop out rate in the later stages, presumably because I was loosing higher level patterns that were already robust and drawn from multiple signals.

I also included an L2 regularizer to stop the weights from getting too large, again to keep the network robust. I found that this lowered my accuracy slightly. However by accident I left a single convolution layer out of the formula (the 6th conv layer, or the second half of the third conv 'block'). This gave me the best results of all so I  left it this way. I could be over fitting my data whereby my actions are influenced by examining the final test score too many times and thereby manually introducing bias by tweaking such a parameter.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
 - Optimizer: Adam
 - Learning Rate: 0.001
 - Number of Epochs: 250
 - Batch Size: 128
 - L2 Loss multiplier: 0.0003
 - First Keep Probability: 0.5
 - Second Keep Probablity: 0.65
 - Third Keep Probability: 0.65

The keep probabilities were all set to 1.0 while testing the model since I didn't want to degrade the test images by dropping signals during testing. I also set the L2 loss multiplier to 0 while training as I didn't want to penalise my test and validation results based on regularisation. I also had to process the images for each batch so that they were in the form required by the augmentation library (which only performs the augmentation as needed to avoid using too much memory). This meant that I then had to reset the images back to the ideal range for the optimiser after augmentation had occurred. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.994
* test set accuracy of 0.98

If an iterative approach was chosen:
* I first tried the LeNet architecture since this was covered in the classes
* With image pre-processing LeNet performed very well but the training accuracy was 1.0 and the validation accuracy in the 90s. This lead me to believe that the model was over fitting. 
* I added drop out layers between the conv layers in the LeNet architecture and an L2 regularizer. This stopped the model from over fitting and I was able to achieve a test accuracy of 0.933
* I wanted to see if I could get an even better accuracy and I tried to implement some of the features of the solution in the LeCunn paper that was linked to in the notebook. I had already used the YUV colour mapping idea from that paper.
* I tried implementing the back to back convolutional layers to handle non-linearity better. and found this improved my results
* I then tried to use augmented data with the back to back convolutional layers and this also improved my results.
* I finally introduced the shortcut architecture to allow explotation of local patterns as well as global and this too improved my results.
* I experimented with different learning rates, optimizers, batch sizes and epochs while working with the LetNet architecture and found the values used to be the best.
* I then adjusted the number of epochs, keep probabilities and L2 regularizer factor to improve the validation score.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

 1. The first image should be easy to classify as it is very clear and without artefacts
 2. The second image has clouds in the background which might confuse the model
 3. The third image is slightly out of the frame and includes a number which is probably harder to classify due to its similarity to other speed limit signs
 4. The fourth sign uses very different colours which the model should be robust to
 5. The fifth sign has a lot of scratches on it and the symbol is quite detailed, which makes it hard to classify when the image is resized to 32x32 pixels.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| General caution       | General caution                               | 
| Right-of-way at the next intersection  | Right-of-way at the next intersection |
| Speed limit (30km/h)  | Speed limit (70km/h)                          |
| Keep right            | Keep right                                    |
| Road work             | Road work                                     |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is a lot lower than the 98% accuracy on the provided test data. But it's also a much smaller sample size. Looking at the softmax probabilities below it also narrowly missed the correct prediction by a couple of percent. I've included the scaled normalized images below and you can see how the model would struggle to tell the difference between 70 and 30km/h. That said, this probably isn't an excuse that would hold up in court when driving 40km over the speed limit!

![alt text][image9]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all the images the model was > 99% sure of it's prediction, expect the one it got wrong. The results for this image are shown below

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .20                   | Speed Limit 70km/h (actual)                   | 
| .17                   | Speed Limit 30km/h (actual)                   |
| .10                   | Stop                                          |
| .10                   | Speed Limit 50km/h (actual)                   |
| .06                   | No Passing                                    |
