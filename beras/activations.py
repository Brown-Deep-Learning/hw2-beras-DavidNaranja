import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):
    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""
        return Tensor(np.where(x > 0, x, x*self.alpha))
        

    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        half = np.where(self.inputs[0] >= 0, 1, self.alpha)
        return [np.where(self.inputs[0] == 0, 0, half)]
    
    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
        
    def forward(self, x) -> Tensor:
        return (1+np.e**(-x))**-1

    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        #(1-sigmoid(x) = dsig)
        sig = (1+np.e**(-self.inputs[0]))**-1
        return [sig*(1-sig)] #More computationally intensive than storing a variable during forward, but I'm just not risking it

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [2470]: Implement for default output activation to bind output to 0-1

    def forward(self, x):
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues


        #Handle each row together, not as individual scalers

        exps = np.exp(x-np.max(x))
        return exps / np.sum(exps, axis=-1, keepdims=True)
        
    

    def get_input_gradients(self):
        """Softmax input gradients!"""
        def softmax_grad_per_vector(x):
            exps = np.exp(x-np.max(x))
            x = exps / np.sum(exps, axis=-1, keepdims=True)
            #x is softmax results

            z = np.outer(x, x) * -1
            np.fill_diagonal(z, x*(1-x)) 
            return Tensor(z)
        
        x = self.inputs[0]
        return [np.apply_along_axis(softmax_grad_per_vector,-1, x)]
    
   