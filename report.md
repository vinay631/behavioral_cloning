**Behavioral Cloning Project**

The objective of this project is to apply deep learning techniques to teach a car how to drive in a simulated environment. The problem can be simplified into predicting steering angles based on images taken from camera mounted to the car in the simulation. These images are taken from three different angles - left, center, right.

Training data is collected by driving the car in test tracks. This training data is then used to train a model based on NVIDIA architecture. 

[//]: # "Image References"

[image1]: ./imgs/steer1.png "Steering angles before augmentation"
[image2]: ./imgs/steer2.png "Steering angles after augmentation"
#### 1. File included

My project includes the following files:
* model.py containing the script to create and train the model
* utils.py contains the preprocessing and data augmentation code
* drive.py for driving the car in autonomous mode
* model.json contains trained convolution neural network
* model.h5 containing the weights of the netowrk
* report.md summarizing the results
* result.md the video of autonomous driving

#### 2. Training and testing the model
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

The model can be trained by using the following:
```sh
python model.py -d data_folder
```


####3. The model code

The model.py file contains the code for training and saving the convolution neural network. The main method has the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

### Model Architecture and Training Strategy

####1. Final model architecture

The model I used is based on NVIDIA end-to-end neural network with a slight modification. The model network (model.py lines 48-61) consists of 5 layers of convolutions with either 3x3 or 5x5 filters. We have a dropout layer followed by fully connected layers. ELU was used in the model to introduce nonlinearity which in turn would help the model generalize better. ELU was chosen over RELU because learning with ELU is shown to be faster than RELU. I also a lambda layer to normalize the input images. 

Here is the final architecture which was coded using Keras:

| Layer (type)                    | Output Shape       | Connected to    |
| ------------------------------- | ------------------ | --------------- |
| lambda_1 (Lambda)               | (None, 66, 200, 3) | lambda_input_1  |
| convolution2d_1 (Convolution2D) | (None, 31, 98, 24) | lambda_1        |
| convolution2d_2 (Convolution2D) | (None, 14, 47, 36) | convolution2d_1 |
| convolution2d_3 (Convolution2D) | (None, 5, 22, 48)  | convolution2d_2 |
| convolution2d_4 (Convolution2D) | (None, 3, 20, 64)  | convolution2d_3 |
| convolution2d_5 (Convolution2D) | (None, 1, 18, 64)  | convolution2d_4 |
| dropout_1 (Dropout)             | (None, 1, 18, 64)  | convolution2d_5 |
| flatten_1 (Flatten)             | (None, 1152)       | dropout_1       |
| dense_1 (Dense)                 | (None, 100)        | flatten_1       |
| dense_2 (Dense)                 | (None, 50)         | dense_1         |
| dense_3 (Dense)                 | (None, 10)         | dense_2         |
| dense_4 (Dense)                 | (None, 1)          | dense_3         |




####2. Data Collection and preprocessing

The data was collected by driving multiple laps in the first track (the "lake track"). Visualizing the steering angles, we can see that the  dataset is highly imbalanced, which means that the predicted angle will most of the time be zero.

![alt text][image1]

So to alleviate this problem, I implementation some data augmentation techniques:

- Randomly choose left, right or center image.
- If left image is selected, +0.2 is added to the steering angle
- If right image is selected, -0.2 is added to the steering angle.
- The images are randomly flipped with a probability of 50%.
- The images are translated vertically and horizontally (steering angle is also adjusted for this)
- The brightness of images were also altered with random amount.

This results in the following steering angle distributions:
![alt text][image2]

With this data augmentation step, the model performed much better. A few image processing steps were also done:

- Image cropping to remove the front of the car and other artifacts like sky etc.
- Images were resized to 66x200 (the size used by NVIDIA model).
- Images were converted from RGB to YUV.

The data augmentation and preprocessing methods are in utils.py in methods "augment" and "preprocess_img" respectively.

I also added some recovery data by driving to the edge of the track and recording the process of bringing the car back to the center. This really helped me avoid the cases where the model did not know what it was supposed to be do when the car approached the edges of the track.

####3. Training the model

The data was split into training and validation dataset using generator method called 'generate_data'. For each epoch, 80% of the data was training data and the 20% data was for validation.

Mean squared error was used for loss function to measure the accuracy of predicted steering angle. Adam optimizer with default learning rate of 1.0e-4 was used. I also added a dropout layer to alleviate overfitting. I used 10 epochs to train the model. I tried going higher but my GPU memory was not enough to support it. 

The code for building and training model can be found in build_model() and train() methods respectively.

####4. Testing the model

I used the first track to test the model. In the beginning, without the recovery data, the car skidded off the track to lake. So the tip to add recovery data proved crucial to the success of the model. The result an be found in ["result.mp4"](result.mp4)

####5. Conclusion and Discussion

This project really helped me grasp the idea of having a balanced dataset for successful model. More than anything else (parameter tuning), only the change in the makeup of the dataset really determined the accuracy of the model. There is a plenty of room for improvement here like using more dataset from the other track and successfully using the model to navigate the second track, trying out different models etc. I will definitely be revisiting these improvements soon. I would also like to "visualize" what the model bases its decision on. Overall though, implementing this project has been very a great learning experience.
