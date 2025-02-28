from collections import defaultdict
import numpy as np

class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def apply_gradients(self, trainable_params, grads):
        for i in range(len(grads)):
            trainable_params[i].assign(trainable_params[i] - self.learning_rate*grads[i])

class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, trainable_params, grads):
        for i in range(len(grads)):
            
            self.v[i] = (1-self.beta)*(grads[i]**2) + self.beta*self.v[i]
            tmp = self.learning_rate/((self.v[i]**.5) + self.epsilon) * grads[i]
            trainable_params[i].assign(trainable_params[i]-tmp)


class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):


        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.t = 0                              # Time counter

    def apply_gradients(self, trainable_params, grads):
        self.t += 1
        for i in range(len(grads)):
            self.m[i] = self.beta_1*self.m[i] + (1-self.beta_1) * grads[i]
            self.v[i] = self.beta_2*self.v[i] + (1-self.beta_2) * grads[i]**2
            mh = self.m[i]/(1-self.beta_1**self.t)
            vh = self.v[i]/(1-self.beta_2**self.t)

            trainable_params[i].assign(trainable_params[i] - self.learning_rate*mh/(vh**.5 + self.epsilon))