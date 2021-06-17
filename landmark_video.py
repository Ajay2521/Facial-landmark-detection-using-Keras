# Import required libraries for this section

# magical function which is used to display visualization in notebook 
# %matplotlib inline

# numpy - used for manipulating array/matrix 
import numpy as np

# matplotlib.pyplot - used for data visualization
import matplotlib.pyplot as plt


# OpenCV library for computer vision - image processing
import cv2                     


# load_model - is used to the load the pre trained model
from keras.models import load_model 

# Facial keypoints (also called facial landmarks) are the small blue-green dots shown on each of the faces in the image above 
# - there are 5 keypoints marked in each image. 
# They mark important areas of the face - the eyes, corners of the mouth, the nose, etc. 
# Facial keypoints can be used in a variety of machine learning applications from face and emotion recognition to commercial applications like the image filters popularized by Snapchat.
# facial keypoint detection is a regression problem.

# A single face corresponds to a set of 5 facial keypoints (a set of 5 corresponding  (𝑥,𝑦)  coordinates, i.e., an output point). 
# Because our input data are images, we can employ a convolutional neural network to recognize patterns in our images 
# and learn how to identify these keypoint given sets of labeled data.

model =  load_model('./model')
# print(model) -> <keras.engine.sequential.Sequential object at 0x7f2a081955d0>
# print(model.input_shape) ->(None, 96, 96, 1)
# print(model.output_shape) -> (None, 10)

# function used to detect the faces and keypoints
def detect_keypoints(image):
  
  # Load in color image for face detection
  # imread - read the image and returns the image in the form of pixel values of array
  image = cv2.imread(image)
  # print('\nImage data :',image) # value in the form of pixel of array
  # print("\nImage Data :",image.shape) # image size -> (665, 1000, 3) -> (n,n,nc)
  # print("\nImage :\n",plt.imshow(image)) #->BGR image  with size (665, 1000, 3)
  
  # Convert the image to RGB chanels
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # print('\nImage data :',image) # value in the form of pixel of array
  # print("\nImage Data :",image.shape) # image size -> (1526, 1800, 3) -> (n,n,nc)
  # print("\nImage :\n",plt.imshow(image)) #->RGB image  with size (1526, 1800, 3)
  
  # Convert the RGB image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  # print('\nGray Image data :',gray_image) # value in the form of pixel of array
  # print("\nGray Image Data :",gray_image.shape) # image size -> (665, 1000) -> (n,n)
  # print("\nGray Image :\n",plt.imshow(gray_image)) #->gray scale image  with size (665, 1000)
  
  # Extract the pre-trained face detector from an xml file
  face_cascade = cv2.CascadeClassifier('detectors/haarcascade_frontalface_default.xml')
  # print(face_cascade) -> <CascadeClassifier 0x7f63849e7590>
  
  # Detect the faces in gray scale image
  # detectMultiScale -  function will return a rectangle with coordinates(x,y,w,h) around the detected face. 
  # It takes 3 common arguments — the input image, scaleFactor, and minNeighbours
  # scaleFactor - specifying how much the image size is reduced at each image scale.
  # minNeighbors – specifying how many neighbors each candidate rectangle should have to retain it.
  faces = face_cascade.detectMultiScale(gray_image)
  
  # returns x axis, y axis, width and heidht of each faces detected using the detectMultiScale
  # The first two entries in the array (extracted in the above code as x and y) specify the horizontal and vertical positions of the top left corner of the bounding box. 
  # The last two entries in the array (extracted here as w and h) specify the width and height of the box.
  # print(faces)  -> [x,y,w,h] - [1295   94   96   96]
  
  # Print the number of faces detected in the image
  # print('\nNumber of faces detected:', len(faces))
  
  # Make a copy of the orginal image to draw face detections on
  detected_image = np.copy(image)
  
  # storing the faces and there keypoints
  num_face_keypoints = []
  
  # Get the bounding box for each detected face
  for (x,y,w,h) in faces:
    
    # getting the roi image
    face_img = detected_image[y:y+h, x:x+w, :]
    # print(face_img) -> [235 208 215]
    # print(face_img.shape)-> (110, 110, 3), the value will vary for each faces detected

    # Pre-process for feeding the data into the model according to the model input
    face_reshaped = cv2.resize(face_img, (96, 96))
    # print(face_reshaped) # -> [235 208 215]
    # print(face_reshaped.shape ) # ->(96, 96, 3), the value will remain same for all the faces detected

    # Convert the RGB image to grayscal
    gray_image = cv2.cvtColor(face_reshaped, cv2.COLOR_RGB2GRAY)
    # print('\nGray Image data :',gray_image) # value in the form of pixel of array, eg- [217 214 213 ...   0   0   1]
    # print("\nGray Image Data :",gray_image.shape) # image size -> (96, 96) -> (n,n)
    # print("\nGray Image :\n",plt.imshow(gray_image)) #->gray scale image  with size (96,96)
    
    # normalizing the pixel value
    gray_normalized = gray_image / 255.
    # print('\ngray_normalized data :',gray_normalized) # value in the form of pixel of array, eg- [0.85098039 0.83921569 0.83529412 ... 0.         0.         0.00392157]
    # print("\ngray_normalized shape :",gray_normalized.shape) # image size -> (96, 96) -> (n,n)
    
    # newaxis - used to increase the dimension of the existing array by one more dimension, when used once. Thus,. 1D array will become 2D
    gray_normalized = gray_normalized[np.newaxis, :, :, np.newaxis]
    # print('\ngray_normalized data :',gray_normalized) # value in the form of pixel of array, eg- [[0.85098039] \n [0.83921569].... \n  [0.        ] \n  [0.00392157]]
    # print("\ngray_normalized shape :",gray_normalized.shape) # image size -> (96, 96) -> (n,n)
    
    # Predicting the keypoints from the model
    key_points = model.predict(gray_normalized)
    # print('\nkey points data:',key_points) # keypoints or features predicted from the model
    # [[ 0.32963058 -0.29287452 -0.3048896  -0.11686236 -0.00556943  0.17257921 0.50848895  0.46883777 -0.20279701  0.644071  ]]
    # eg [[x1, y1, x2, y3 ... xn, yn]]
    # print('\nkey points shape:',key_points.shape) -> (1, 10)
    
    # re-scaling the normalized keypoints value 
    key_points =( key_points * 48) + 48
    # print('\nkey points data:',key_points) # keypoints or features predicted from the model
    # [[63.822266 33.942024 33.3653   42.390606 47.732666 56.283802 72.40747 70.50421  38.265743 78.915405]]]]
    # eg [[x1, y1, x2, y3 ... xn, yn]]
    # print('\nkey points shape:',key_points.shape)#  -> (1, 10)
    

    # skipping up the one value which is basically a y coords
    x_coords = key_points[0][0::2] # -> [x1, x2...xn]
    # print('\nx_coords data:',x_coords) # getting x coords from the keypoint
    # [63.822266 33.3653   47.732666 72.40747  38.265743]
    # print('\nx_coords shape:',x_coords.shape)#  -> (5, )
    
    # skipping up the one value which is basically a x coords
    y_coords = key_points[0][1::2] # -> [y1, y2...yn]
    # print('\ny_coords data:',y_coords) # getting y coords from the keypoint
    # [33.942024 42.390606 56.283802 70.50421  78.915405]
    # print('\ny_coords shape:',y_coords.shape)#  -> (5, )
    
    # re-normaling the coordinates
    x_coords = x_coords * w / 96 + x
    # print('\nx_coords data:',x_coords) # getting x coords from the keypoint
    # [646.1297  611.2311  627.69366 655.9669  616.8462 ]
    # print('\nx_coords shape:',x_coords.shape)#  -> (5, )
    
    y_coords = y_coords * h / 96 + y
    # print('\ny_coords data:',y_coords) # getting y coords from the keypoint
    # [ 84.89191  94.57257 110.49186 126.78608 136.42389]
    # print('\ny_coords shape:',y_coords.shape)#  -> (5, )
    
    num_face_keypoints.append((x_coords, y_coords))
    #print('\nnum_face_keypoints data:' ,num_face_keypoints) # getting x,y coords from the num_face_keypoints
    # [(array([646.1297 , 611.2311 , 627.69366, 655.9669 , 616.8462 ],dtype=float32),
    #  array([ 84.89191,  94.57257, 110.49186, 126.78608, 136.42389],dtype=float32))]

  return num_face_keypoints, detected_image

# defining the image path
image = 'test3.jpg'

# getting the landmarks and the detcted image
keypoints, detected_image = detect_keypoints(image)

# marking the circle to the landmarks using the x and y coordinates of each faces
for face in keypoints:
    
    # getting the x and y coordinates from the landmarks or keypoints  
    for x, y in zip(face[0], face[1]):
        cv2.circle(detected_image, (x, y), 3, (0,255,0),-1)

# Convert the image to RGB chanels
detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
# print('\nImage data :',image) # value in the form of pixel of array
# print("\nImage Data :",image.shape) # image size -> (96,96, 3) -> (n,n,nc)
# print("\nImage :\n",plt.imshow(image)) #->RGB image  with size (96,96, 3)

cv2.imwrite("landmarked_image.jpg",detected_image)

