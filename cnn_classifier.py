from data_parser import getData

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
import os
import numpy as np
import random
from keras.metrics import MeanSquaredError, AUC, BinaryAccuracy 
from calculate_metrics import get_metrics

import matplotlib.pyplot as plt

EPOCHS = 300
BATCH_SIZE = 16
LOADED = False

def binary_labels(x):
    bin = []
    for ex in x:
        if ex >= .5: bin.append(1) 
        else: bin.append(0)
    return bin

def buildModel(x_train, y_train, x_validation, y_validation):
    
    model = Sequential()
    
    # convolutional layers
    model.add(Conv2D(32,2,activation='relu', input_shape=(32,32,1)))
    model.add(MaxPool2D(2))
    model.add(Conv2D(64,2,activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(128,2,activation='relu'))
    model.add(MaxPool2D(2))

    model.add(Flatten())
    model.add(Dense(16))
    #model.add(Dense(1))
    model.add(Dense(1, activation='sigmoid')) # for categorical
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            MeanSquaredError(),
            AUC(),
            BinaryAccuracy()
        ]
)

    print("Training...")
    # training the model for 10 epochs
 
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_validation, y_validation))
    #model.save("weights.hdf5")
    #weights = model.get_weights() 
    #print(weights)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylim((0,5.5))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('train_vs_testing_loss_300_volume_binary.png')

    plt.show()

    return model

def testModel(model, x_test, y_test):
    print("Testing...")
    y_pred_train = model.predict(x_train, verbose = 1)
    y_pred_test = model.predict(x_test, verbose = 1)
    return y_pred_train, y_pred_test
    


x_train, y_train, x_validation, y_validation, x_test, y_test = getData(LOADED)
print("training set x: ", x_train.shape, "training set y: ",y_train.shape)
print("validation set x:", x_validation.shape, "validation set y: ", y_validation.shape)
print("test set x: ", x_test.shape, "test set y: ",y_test.shape)


model = buildModel(x_train, y_train, x_validation, y_validation)
y_pred_train, y_pred_test = testModel(model, x_test, y_test)
y_pred_train = binary_labels(y_pred_train)
y_pred_test = binary_labels(y_pred_test)
get_metrics(y_test, y_pred_test)

#print(y_pred_train)
plot_1 = plt.plot(y_pred_train, y_train,'.',color = 'rosybrown', markersize=8)
plt.ylabel('y_train')
plt.xlabel('y_prediction')
plt.savefig('train_300_epochs_volume_binary.png')
plt.show()


plt.plot(y_pred_test, y_test, '.',color = 'rosybrown', markersize=8)
plt.ylabel('y_test')
plt.xlabel('pred')
plt.savefig('test_300_epochs_volume_binary.png')
plt.show()

with open('output_100_volume_binary.txt', 'w') as filehandle:
    for listitem in y_pred_test:
        filehandle.write('%s\n' % listitem)


