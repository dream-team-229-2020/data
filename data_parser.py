import os
import pickle
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np

def binarize(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] > .05: img[i][j][0] = 1
    return img
    

def binary_labels(x):
    bin = []
    for ex in x:
        if ex >= 0: bin.append(1) 
        else: bin.append(0)
    return bin

def extractData():
    prediction_type = 'volume_bar' # can change to down_bars up_bars volume_bars
    label_type = 'percentages.txt' # can also be strict_increase_labels.txt or monotonic_increase_labels.txt

    base_dir = os.getcwd()


    #img_dir = os.path.join(base_dir,'new_dataset', prediction_type)
    img_dir = os.path.join(base_dir,'new_dataset', prediction_type)
    n = len(os.listdir(img_dir)) #number of examples
    num_train = int(.7*n) # change to proportion
    num_validation = int(.15*n)
    num_test = int(.15*n)

    x_data = []
    for i in range(n):
        img_name = str(i) + '.png'
        img_path = os.path.join(img_dir,img_name)
        img = load_img(img_path, target_size=(32, 32), color_mode = 'grayscale')
        img_array = np.array(img_to_array(img))

        img_array = binarize(img_array)
        x_data.append(img_array)
        print(i)

    #imgage_back_test = array_to_img(x_data[0]) #make sure image is being correctly transformed to matrix
    #imgage_back_test.show()
    
    x_train = np.array(x_data[:num_train])
    x_validation = np.array(x_data[num_train:num_validation + num_train])
    x_test = np.array(x_data[num_validation + num_train:])

    labels = np.genfromtxt(os.path.join(base_dir,label_type),delimiter=',')
    labels = binary_labels(np.reshape(labels, (len(labels),))) #if categorical

    y_train = np.array(labels[:num_train])
    y_validation = np.array(labels[num_train:num_validation + num_train])
    y_test = np.array(labels[num_validation + num_train:])
   

    return x_train, y_train, x_validation, y_validation, x_test, y_test




def getData(loaded):
    print('Parsing...')
    if loaded:
        #training data too big for one 
        
        with open('processed_data.npy', 'rb') as f:
            x_train = np.load(f)
            y_train = np.load(f)
            x_validation = np.load(f)
            y_validation = np.load(f)
            x_test = np.load(f)
            y_test = np.load(f)
        
    else:
        x_train, y_train, x_validation, y_validation, x_test, y_test = extractData()
        
        with open('processed_data.npy', 'wb') as f:
            np.save(f, x_train)
            np.save(f, y_train)
            np.save(f, x_validation)
            np.save(f, y_validation)
            np.save(f, x_test)
            np.save(f, y_test)
        
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test
