import pandas as pd
import numpy as np
import gc
import random

from data_parser import load_data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization 


from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16


def create_model(train_generator, validation_generator):
    vgg_base = VGG16(weights='imagenet',    # use weights for ImageNet
                    include_top=False,     # drop the Dense layers!
                    input_shape=(150, 150, 3))

    model = Sequential([
        # our vgg16_base model added as a layer
        vgg_base,
        # here is our custom prediction layer (same as before)
        Flatten(),
        Dropout(0.50),
        Dense(1024, activation='relu'),
        Dropout(0.20),        
        Dense(512, activation='relu'),
        Dropout(0.10),         
        Dense(1, activation='sigmoid')    
    ])
    
    # mark vgg_base as non-trainable, so training updates
    # weights and biases of just our newly added layers
    vgg_base.trainable = False
    
    model.compile(optimizer=Adam(lr=1e-4), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_generator,
        steps_per_epoch=709,
        epochs=150,
        validation_data=validation_generator,
        validation_steps=71)
        





















'''
def create_model(train_generator, validation_generator):
    
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    sgd = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    filepath = "final_weights.hdf5"

    checkpoint = ModelCheckpoint( # used for training loss
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    model_callbacks = [checkpoint]

    history = model.fit(train_generator,
      steps_per_epoch=100,  
      epochs=15,
      verbose=1,
      #validation_data = validation_generator,
      #validation_steps=10, 
      callbacks=model_callbacks)

    model.evaluate(validation_generator)
 

#training_input, training_output, testing_input, testing_output = load_data(False)
#print(training_input.shape, training_output.shape, testing_input.shape, testing_output.shape)
'''


train_generator, validation_generator = load_data()
create_model(train_generator, validation_generator)
