import numpy as np

from beras.onehot import OneHotEncoder
from beras.core import Tensor
from tensorflow.keras import datasets

def load_and_preprocess_data() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    '''This is where we load in and preprocess our data! We load in the data 
        for you but you'll need to flatten the images, normalize the values and 
        convert the input images from numpy arrays into tensors
    Return the preprocessed training and testing data and labels!'''
    
    #Load in the training and testing data from the MNIST dataset
    (train_inputs, train_labels), (test_inputs, test_labels) = datasets.mnist.load_data()
    #max = np.max(train_inputs)
    train_inputs = (train_inputs/255.).astype(np.float32)
    max = np.max(test_inputs)
    test_inputs = (test_inputs/255.).astype(np.float32)

    #print(train_inputs.shape)
    #z = train_inputs[0]
    train_inputs = np.reshape(train_inputs, (-1, 28**2))
    test_inputs = np.reshape(test_inputs, (-1, 28**2))

    train_inputs = Tensor(train_inputs)
    test_inputs = Tensor(test_inputs)
    #print(train_inputs.shape)
    '''zz = train_inputs[0]
    print(z)
    print(zz)
    zz = np.reshape(zz, (28, -1))
    print(z == zz)'''
    #Commented out dumb proof that it got reshaped properly
    return (train_inputs, train_labels, test_inputs, test_labels)
    

load_and_preprocess_data()