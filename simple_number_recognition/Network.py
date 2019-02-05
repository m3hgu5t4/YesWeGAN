# copied from https://m.youtube.com/watch?v=aircAruvnKk, http://neuralnetworksanddeeplearning.com/
import numpy as np

class Network(object):
    def __init__(self, *args, **kwargs):
        pass

    def feedforward(self, a):
        """Return the output of the network for an input vector a"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a