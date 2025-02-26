import numpy as np

from beras.core import Callable
#from core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='
        mask = np.argmax(probs, -1) == np.argmax(labels, -1)
        return np.mean(mask)
    

'''from core import Tensor
c = CategoricalAccuracy()
tprobs = Tensor([[0, 1, 0],
                 [1, 0, 0],
                 [.2, 0, .8],
                 [0, .8, .2],
                 [.2, .3, .3]])

tlabels = Tensor([[0, 1, 0],
                 [1, 0, 0],
                 [1, 0, 0],
                 [0, 1, 0],
                 [0, 0,1]])
print(c(tprobs, tlabels))'''
