from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import os
import numpy as np

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

import matplotlib.pyplot as plt
 


def turn_binary(labels):
    binary = []
    for label in labels:
        if label > 0:
            binary.append(1)
        else: binary.append(0)
    return binary

print("Parsing...")
base_dir = os.getcwd()
image_dir = os.path.join(base_dir,'all_images')
images = os.listdir('all_images')
num_train = 1600
#print("We have ", num_train, " training images and ", len(images) - 1 - num_train, "testing images")

x_data = []
for i in range(len(images)-1):
    img_name = str(i) + '.png'
    img_path = os.path.join(image_dir,img_name)
    img = load_img(img_path, target_size=(108, 105))
    img_array = img_to_array(img)
    x_data.append(img_array)

x_train = np.array(x_data[:num_train])
x_test = np.array(x_data[num_train:])

labels = np.genfromtxt(os.path.join(base_dir,'all_percentages.csv'),delimiter=',')[1:,1:]
#labels = turn_binary(np.reshape(labels, (len(labels),)))
y_train = np.array(labels[:num_train])
y_test = np.array(labels[num_train:])

print("training set x: ", x_train.shape, "training set y: ",y_train.shape)
print("test set x: ", x_test.shape, "test set y: ",y_test.shape)

#_train = np_utils.to_categorical(y_train, 2)
#y_test = np_utils.to_categorical(y_test, 2)




# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(108,105,3)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(1))
#model.add(Dense(2, activation = 'softmax'))


# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

print("Training...")
# training the model for 10 epochs
print("Testing...")
model.fit(x_train, y_train, batch_size=30, epochs=10)

y_pred = model.predict(x_test)
print("predictions", y_pred)
print("labels", y_test)
print(len(y_pred), len(y_test))

plt.plot(y_test, y_pred, 'ro')
plt.ylabel('test actual vs prediction')
plt.show()

