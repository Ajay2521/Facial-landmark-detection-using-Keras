# Import required libraries for this section

# magical function which is used to display visualization in notebook 
# %matplotlib inline

# numpy - used for manipulating array/matrix 
import numpy as np

# matplotlib.pyplot - used for data visualization
import matplotlib.pyplot as plt

# math - used for basic math operation
# import math

# OpenCV library for computer vision - image processing
import cv2                     

# PIL - Python Imaging Library
# from PIL import Image

# time - used for time related operation
# import time 

# alternative for cv2.imshow, since colab block cv2.imshow functionality due to kernel kill
# from google.colab.patches import cv2_imshow

# pandas - for data accessing and manipulation
import pandas as pd

import tensorflow as tf

# shuffle - for shuffling the datas
from sklearn.utils import shuffle

# Sequential - A base layer where the extra layer of the network can be added
from keras.models import Sequential

# Convolution2D -layer used to perform the convolution operationel 
from keras.layers import Convolution2D

# MaxPooling2D - layer used to perform the MaxPooling operation
from keras.layers import MaxPooling2D

# Dropout - Layer used to drop the ddata which can lea to over fitting of the mod
from keras.layers import Dropout

# Flatten - layer which is used to Flatten a 2D array value into 1D array/features values
from keras.layers import Flatten

# Dense - Layer which is used to make a fully connected layer
from keras.layers import Dense

# Adam - Optimiser which is used to optimise the model
from keras.optimizers import Adam

# ModelCheckpoint - Chekpoint where the model is saved based on best values
from keras.callbacks import ModelCheckpoint 

def load_data(test=False):
    """
    Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Important that the files are in a `data` directory
    """  

    # defining the test and training dataset path
    FTRAIN = 'data/training.csv'
    # print("\nFTRAIN :\n",FTRAIN)-> data/training.csv
    
    FTEST = 'data/test.csv'
    # print("\nFTEST :\n",FTEST)-> data/test.csv
    
    fname = FTEST if test else FTRAIN
    # print(fname) data/training.csv and data/test.csv

    # reading the csv file
    # load dataframes
    df = pd.read_csv(os.path.expanduser(fname))
    
    # print(df.shape)  # (1783, 2)

    # The Image column has pixel values separated by space; 
    # convert the values to numpy arrays:
    # print(df['Image']) -> 238 , Name: Image, Length: 7049, dtype: object
    # fromstring - function is used to create a new 1-D array initialized from raw binary or text data in a string
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' '))
    # print(df['Image']) -> [238.0] , Name: Image, Length: 7049, dtype: object
    
    # drop all rows that have missing values in them
    df = df.dropna() 
    
    # scale pixel values to [0, 1]
    # np.vstack -  used to stack the sequence of input arrays vertically to make a single array
    x = np.vstack(df['Image'].values) / 255.
    # print(x)  -> [0.93333333]
    # print(x.dtype) -> float64

    # changing the datatype
    x = x.astype(np.float32)
    # print(x) - >[0.93333334]
    # print(x.dtype) -> float32
    
    # changing the shape
    x = x.reshape(-1, 96, 96, 1) 
    # print(x.shape)  -> return each images as 96 x 96 x 1

    # only FTRAIN has target columns

    if not test:  
       
        # post processing to make the 15 landmarks into 5 landmarks
        df.drop(['left_eye_inner_corner_x',
                 'left_eye_inner_corner_y',
                 'left_eye_outer_corner_x',
                 'left_eye_outer_corner_y',
                 'right_eye_inner_corner_x',
                 'right_eye_inner_corner_y',
                 'right_eye_outer_corner_x',
                 'right_eye_outer_corner_y',
                 'left_eyebrow_inner_end_x',
                 'left_eyebrow_inner_end_y',
                 'left_eyebrow_outer_end_x',
                 'left_eyebrow_outer_end_y',
                 'right_eyebrow_inner_end_x',
                 'right_eyebrow_inner_end_y',
                 'right_eyebrow_outer_end_x',
                 'right_eyebrow_outer_end_y',
                 'mouth_center_top_lip_x',
                 'mouth_center_top_lip_y',
                 'mouth_center_bottom_lip_x',
                 'mouth_center_bottom_lip_y' ], axis = 1, inplace = True)

        
        # getting the target data and converting it into array
        y = df[df.columns[:-1]].values
        # print(y) ->[66.03356391 39.00227368 30.22700752 ... 79.97016541 28.61449624 77.38899248]
        # print(y.shape) -> (2140, 10)
        
        # scale / normalizing target coordinates to [-1, 1]
        y = (y - 48) / 48
        # print(y)  ->[ 0.37569925 -0.18745263 -0.37027068 ...  0.66604511 -0.40386466 0.61227068]
        # print(y.shape) # -> (2140, 10)
        
        # shuffle train data
        x, y = shuffle(x, y, random_state=42)
        # print(x,y)

        # changing the datatype
        y = y.astype(np.float32)
        # print(y) #- >[ 0.3816111  -0.21757638 -0.40208334 ...  0.5116389  -0.38531944  0.5158264 ]
        # print(y.dtype) # -> float32
    
    else:
        
        y = None

    return x, y



# Load training set
x_train, y_train = load_data()
# print(x_train) -> [0.79607844], value in the form of numpy array
# print("x_train.shape ==",x_train.shape) # (2140, 96, 96, 1)
# print(y_train) ->[ 0.3816111  -0.21757638 -0.40208334 ...  0.5116389  -0.38531944, 0.5158264 ] ,value in the form of numpy array
# print("y_train.shape ==", y_train.shape) # (2140,10)
    
# Load testing set
x_test, _ = load_data(test=True)
# print("x_test.shape ==",x_test.shape) # (1783, 96, 96, 1)
# print(x_test) -> [0.7137255 ], value in the form of numpy array

# Import deep learning resources from Keras
# model accept 96x96 pixel graysale images in
# It should have a fully-connected output layer with 10 values (2 for each facial keypoint)

# A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
model = Sequential()

# Convolution2D - convolution layer
# 64 - no . of filters used
# 3, 3, - size of the filter/ kernel
# input shape - input of the model
# activation - non linear function used for checking the active status of the neuron
model.add(Convolution2D(64, 3, 3, input_shape=(x_train.shape[1:])))
model.add(Convolution2D(64, 3, 3, activation='relu'))

# MaxPooling2D - max polling used to get the maxmium features out
model.add(MaxPooling2D())

# Flattern - Converting the 2D array values into 1D flattern data
model.add(Flatten())

# Dense - Dense layer with 128 features
model.add(Dense(128, activation='tanh'))

# Dropout - Dropout the data to make sure the moel doesnt over fit
model.add(Dropout(0.3))

# Dense - Dense layer with 30 features
model.add(Dense(10, activation='tanh'))


# Summarize the model
model.summary()

# Total params: 246,366
# Trainable params: 246,366

# Compiling the model
# loss = used to compute the quantity that a model should seek to minimize during training.
# landmark detection is a Regression losses problem thus using mean_squared_error
# mean_squared_error - Computes the mean of squares of errors between labels and predictions.
# optimizer - is used to optimize the moddel for the loss value to achive the global minima
# adam - adam is a optimiser function which is stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
model.compile(loss='mean_squared_error', optimizer='adam')

# Save the model as model.h5
# ModelCheckpoint : used in conjunction with training using model.fit() to save a model or weights (in a checkpoint file) at some interval
# filepath: string or PathLike, path to save the model file
# save_best_only=True - only saves when the model is considered the "best"
# save_weigths_only=True - only saves when the model is considered the "best"
# verbose: verbosity mode, 0 or 1.
# verbose = 1, which includes both progress bar and one line per epoch. verbose = 0, means silent
# verbose - helps to detect overfitting which occurs if your acc keeps improving while your val_acc gets worse.
checkpointer = ModelCheckpoint(filepath='model/model.h5',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True)

# training the model
# x_train, y_train -input datas
# batch_size - no of batches to be performed
# epochs - no of epochs to be done for all the batches
# validation_split - spliting the data for validation, which can be used to validate the model
# callbacks -  used to monitor your metrics 
# verbose: verbosity mode, 0 or 1.
# verbose = 1, which includes both progress bar and one line per epoch. verbose = 0, means silent
# verbose - helps to detect overfitting which occurs if your acc keeps improving while your val_acc gets worse.
# shuffle - used to shuffle the data for random access
hist = model.fit(x_train, y_train,
                 batch_size=64,
                 epochs=30,
                 validation_split=0.2,
                 callbacks=[checkpointer], 
                 verbose=1,
                 shuffle=True)

# saving the model with .pb format
tf.saved_model.save(model,'model')