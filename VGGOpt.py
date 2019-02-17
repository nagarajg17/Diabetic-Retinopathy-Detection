import pandas as pd
import numpy as np
import os
import keras
from graph import plotTrainingGraph
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

#original VGG16 network weight file
#model1 = VGG16(include_top = True, weights = 'imagenet', classes = 1000)  
#loading weights without fc and softmax layers for Transfer learning.
model = VGG16(include_top = False, weights = 'imagenet', input_shape = (224,224,3))
flatten_model = Sequential()
flatten_model.add(model)
flatten_model.add(Flatten())

#flatten the last max pool layer outputs
features_input = flatten_model.predict(img_arr)
#print(model.summary())
#print(model.inputs)
#print(model.outputs)



new_model = Sequential()

#add a fc1 layer with 4096 output neurons and a dropout layer1
new_model.add(units = 4096, activation = 'relu', kernel_initializer='uniform')
new_model.add(keras.layers.Dropout(rate = 0.5, noise_shape=None, seed=None))

#add a fc2 layer with 4096 output neurons and a dropout layer2
new_model.add(units = 4096, activation = 'relu', kernel_initializer='uniform')
new_model.add(keras.layers.Dropout(rate = 0.5, noise_shape=None, seed=None))

#adding a softmax layer with 5 classes
new_model.add(Dense(units = 5))
new_model.add(keras.layers.Activation('softmax'))

#configures the model for training
sgd = SGD(lr=0.001, decay=0.0, momentum=0.9,nesterov=True)
new_model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'] )

#train the model for given number of epochs
training_history = {}
try:
    training_history = new_model.fit(x = features_input, y = labels, batch_size = , epochs = , verbose = 2, validation_split = 0.3)
except Exception as error:
    print error

print(training_history.history)   #print loss and accuracy of training and validation data
plotTrainingGraph(training_history) #plot acc and loss graph for training and validation dataset
new_model.save('myModel.h5')         #saving the model
new_model.save_weights('myModelWeights.h5') #saving trained weight file

#return accuracy for the test data with trained weights
loss = 0
accuracy = 0
try:
    loss , accuracy = new_model.evaluate(x=, y=, batch_size=, verbose=1)
except Exception as error:
    print error

print("The testing data loss is ", loss)
print("The testing data accuracy is ", accuracy)





