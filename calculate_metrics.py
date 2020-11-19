import numpy as np
import os 
import matplotlib.pyplot as plt


base_dir = os.getcwd()
label_type = 'percentages.txt'
prediction_type = 'output_300_down.txt' #can be lots of things

def binary_labels(x):
    bin = []
    for ex in x:
        if ex >= 0: bin.append(1) 
        else: bin.append(0)
    print("here")
    return bin
##classification

    #accuracy

    #precision

    #recall

    #confision matrix

##regression

    #average error

    #own metric

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# import some data to play with


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Prediction on Volume data Confusion Matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
   # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Price Prediction Confusion Matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax




def get_rates(label, predictions):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(labels)):
        if label[i] == 1 and predictions[i] == 1:
            tp += 1
        elif label[i] == 0 and predictions[i] == 0:
            tn += 1
        elif label[i] == 0 and predictions[i] == 1:
            fp += 1
        else: fn += 1
    
    return tp,tn,fp,fn


def get_metrics(labels, predictions):
    tp,tn,fp,fn = get_rates(labels, predictions)
    print(tp,tn,fp,fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    print("accuracy:", accuracy, "   precision: ", precision, "   recall: ", recall)

def get_error(labels, predictions):
    difference = labels - predictions
    square = difference ** 2
    return square/len(labels)




'''
get_metrics(labels, predictions)
class_names = ["UP", "Down"]
plot_confusion_matrix(labels, predictions, class_names)
plot_file_name = prediction_type[:-4] + '_confusion_matrix.png'
plt.savefig(plot_file_name)
plt.show()
'''