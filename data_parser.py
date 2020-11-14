import os
import pickle
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np



def extractData():
    prediction_type = 'volume_bars' # can change to down_bars up_bars
    label_type = 'percentages.txt' # can also be strict_increase_labels.txt or monotonic_increase_labels.txt

    base_dir = os.getcwd()


    img_dir = os.path.join(base_dir,'new_dataset', prediction_type)
    n = len(os.listdir(img_dir)) #number of examples

    num_train = int(.7*n) # change to proportion
    num_validation = int(.15*n)
    num_test = int(.15*n)

    #print("using ", num_train, " training examples ", num_validation, " validation examples, and ", num_test, "training examples which should total", n, "but actually totals", num_train + num_validation +  num_test)
    #print("We have ", num_train, " training images and ", len(images) - 1 - num_train, "testing images")

    x_data = []
    for i in range(n-1):
        img_name = str(i) + '.png'
        img_path = os.path.join(img_dir,img_name)
        img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(img)
        x_data.append(img_array)

    x_train = np.array(x_data[:num_train])
    x_validation = np.array(x_data[num_train:num_validation + num_train])
    x_test = np.array(x_data[num_validation + num_train:])


    labels = np.genfromtxt(os.path.join(base_dir,label_type),delimiter=',')

    #labels = turn_binary(np.reshape(labels, (len(labels),))) #if categorical

    y_train = np.array(labels[:num_train])
    y_validation = np.array(labels[num_train:num_validation + num_train])
    y_test = np.array(labels[num_validation + num_train:-1])

    return x_train, y_train, x_validation, y_validation, x_test, y_test




def getData(loaded):
    print('Parsing...')
    if loaded:
        #training data too big for one 
        with open('cnn_input_pickle/x_train_pt1', 'rb') as filepath:
            x_train_pt1 = pickle.load(filepath)
        with open('cnn_input_pickle/x_train_pt2', 'rb') as filepath:
            x_train_pt2 = pickle.load(filepath)
        with open('cnn_input_pickle/y_train', 'rb') as filepath:
            y_train = pickle.load(filepath)
        with open('cnn_input_pickle/x_validation', 'rb') as filepath:
            x_validation = pickle.load(filepath)
        with open('cnn_input_pickle/y_validation', 'rb') as filepath:
            y_validation = pickle.load(filepath)
        with open('cnn_input_pickle/x_test', 'rb') as filepath:
            x_test = pickle.load(filepath)
        with open('cnn_input_pickle/y_test', 'rb') as filepath:
            y_test = pickle.load(filepath)
        x_train = np.concatenate([x_train_pt1,x_train_pt2])
    else:
        x_train, y_train, x_validation, y_validation, x_test, y_test = extractData()
        '''
        with open('cnn_input_pickle/x_train_pt1', 'wb') as filepath:
            pickle.dump(x_train[:int(len(x_train)/2)], filepath)
        with open('cnn_input_pickle/x_train_pt2', 'wb') as filepath:
            pickle.dump(x_train[int(len(x_train)/2):], filepath)
        with open('cnn_input_pickle/y_train', 'wb') as filepath:
            pickle.dump(y_train, filepath)
        with open('cnn_input_pickle/x_validation', 'wb') as filepath:
            pickle.dump(x_validation, filepath)
        with open('cnn_input_pickle/y_validation', 'wb') as filepath:
            pickle.dump(y_validation, filepath)
        with open('cnn_input_pickle/x_test', 'wb') as filepath:
            pickle.dump(x_test, filepath)
        with open('cnn_input_pickle/y_test', 'wb') as filepath:
            pickle.dump(y_test, filepath)
        '''
        
    return x_train, y_train, x_validation, y_validation, x_test, y_test
