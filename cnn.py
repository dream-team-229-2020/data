from data_parser import getData

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import os
import numpy as np

import matplotlib.pyplot as plt
 
loaded = False
x_train, y_train, x_validation, y_validation, x_test, y_test = getData(loaded)
print("training set x: ", x_train.shape, "training set y: ",y_train.shape)
print("validation set x:", x_validation.shape, "validation set y: ", y_validation.shape)
print("test set x: ", x_test.shape, "test set y: ",y_test.shape)


# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(16,3,activation='relu', input_shape=(150,150,3)))
model.add(MaxPool2D(2))
model.add(Conv2D(32,3,activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(64,3,activation='relu'))
model.add(MaxPool2D(2))

# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(512, activation='relu'))
# output layer
model.add(Dense(1))
#model.add(Dense(2, activation = 'softmax')) # for categorical


# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

print("Training...")
# training the model for 10 epochs

model.fit(x_train, y_train, batch_size=30, epochs=5, validation_data=(x_validation, y_validation))

print("Testing...")
y_pred = model.predict(x_test)
#print("predictions", y_pred)
#print("labels", y_test)
#print(len(y_pred), len(y_test))

plt.plot(y_test, y_pred, 'ro')
plt.ylabel('test actual vs prediction')
plt.show()


