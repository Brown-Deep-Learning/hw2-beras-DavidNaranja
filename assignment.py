from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

def get_model():
    model = SequentialModel(
        [
            Dense(784, 200, "kaiming"),
            ReLU(),
            Dense(200, 50, "kaiming"),
            LeakyReLU(),
            Dense(50, 10, "kaiming"),
            Softmax()
           # Add in your layers here as elements of the list!
           # e.g. Dense(10, 10),
        ]
    )
    return model

def get_optimizer():
    # choose an optimizer, initialize it and return it!
    #Choose Adam
    return Adam(0.01)

def get_loss_fn():
    # choose a loss function, initialize it and return it!
    #Choose cross
    return CategoricalCrossEntropy()

def get_acc_fn():
    # choose an accuracy metric, initialize it and return it!
    #Choose categorical? What else?
    return CategoricalAccuracy()

if __name__ == '__main__':
    pass
    ### Use this area to test your implementation!

    # 1. Create a SequentialModel using get_model
    model = get_model()
    # 2. Compile the model with optimizer, loss function, and accuracy metric
    model.compile(get_optimizer(), get_loss_fn(), get_acc_fn())
    # 3. Load and preprocess the data
    tri, trl, tei, tel = load_and_preprocess_data()
    ohe = OneHotEncoder()
    trl = ohe(trl)
    tel = ohe(tel)
    # 4. Train the model
    model.fit(tri, trl, 10, 40)
    # 5. Evaluate the model
    model.evaluate(tei, tel, 100)
    
    
