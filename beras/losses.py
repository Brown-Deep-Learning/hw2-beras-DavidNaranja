import numpy as np

from beras.core import Diffable, Tensor
#from core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        #I'm assuming y_pred is functionally a list of one hot vectors. Loss_raw is difference in results
        # MSE by sample is mean by row to get scalar answer of how that one guess did
        # Output is mean score of those guesses 
        loss_raw = (y_true - y_pred)**2
        MSE_by_sample = np.mean(loss_raw, 1)

        return np.mean(MSE_by_sample)

    def get_input_gradients(self) -> list[Tensor]:
        #With respect to y_pred
        y_pred, y_true = self.inputs[0:2]
        return [(y_pred - y_true), Tensor(1)]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""    
        return np.mean(-1 * np.sum(np.log(y_pred)*y_true, 1))

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs[0:2]
        return [y_true / y_pred * -1, Tensor(1)]

'''
y = Tensor([[0, 1, 0], [0, .7, 0]])
yi = Tensor([[0.1, 0.7, 0.1], [0.1, 0.7, 0.1]])

y = Tensor([[0, 1, 0], [0, 1, 0]])
yi = Tensor([[0, 1, 0], [1, 0, 0]])
c = CategoricalCrossEntropy()
print(c(yi, y))
m = MeanSquaredError()
print(m(yi, y))
#print(c.get_input_gradients())
'''