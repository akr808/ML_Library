# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Convolution Neural Network
@author: Anand
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image

#Data Preprocessing
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)


train_set = train_datagen.flow_from_directory(
    directory="../Images/CNN/training_set",
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
    )
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = train_datagen.flow_from_directory(
    directory="../Images/CNN/test_set",
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
    )

cnn = tf.keras.models.Sequential()

#Adding convolution layer
cnn.add(tf.keras.layers.Conv2D(input_shape = [64,64,3], filters = 32, kernel_size= 3, activation = 'relu'))

#Adding Pooling layers
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Adding second convolution layer & pooling
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size= 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Flattening
cnn.add(tf.keras.layers.Flatten())

#Full Connection to ANN
#Adding the hidden layers
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

#Adding the output layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


cnn.fit(x = train_set, validation_data =test_set, epochs = 25)

#Single prediction
test_image = image.load_img("../Images/CNN/single_prediction/cat_or_dog_2.jpg",target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#Predicting the image
result = cnn.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
  prediction = 'cat'


