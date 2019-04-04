# Car-color-recognition-using-CNN
Vehicule color recognition using CNN and developed with Deeplearning4j library

This is an implementation the car color recognition method described in the paper: "Vehicle Color Recognition using Convolutional
Neural Network" written by Reza Fuad Rachmadi and I Ketut Eddy Purnama. The model is intended to detect vehicule colors from traffic cameras. The model is implemented using Deeplearning4j library.

# Network Architecture

![Untitled](https://user-images.githubusercontent.com/1300982/54075578-4bc6d300-42a1-11e9-891f-ed67a09e6f7e.png)

To simplify, our convolutional network will try to recognize 5 colors (Black, Blue, Grey, Red and White)

# The dataset

The data set used to train and test the model is the publicly vehicle color recognition dataset provided by Chen. But due to the resource limitation issue of the training and testing environment (I am using my Acer ASPIRE 4750G laptop which is endowed with following characteristics: Processor: Intel(R) Core(TM) i3-2310M CPU @ 2.10GHz  2.10GHz; Installer memory (RAM): 4.00 GB), 
just a small portion of the dataset was used (105 images for training and 45 images for testing)

# Results

It took around 1h20min to train the network on my Acer ASPIRE 4750G laptop to produce the following results:

![9](https://user-images.githubusercontent.com/1300982/54075626-e0313580-42a1-11e9-80bc-3b7788926419.png)
