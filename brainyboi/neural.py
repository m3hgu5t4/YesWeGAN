#surorisingly, you only need numpy to
#implement a decent neural network
import numpy as np

#we use the sigmoid activation function defined here
#because it has a very simple derivative
#and is relatively quick to compute
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#define our class
class Net():
    def __init__(self):
        #our input layer
        self.input = np.zeros((784, 1))
        #a1 and b1 are the level 1 weights and biases.
        #they start off randomly initialised
        self.a1 = np.random.random((12, 784)) - 0.5
        self.b1 = np.random.random((12, 1)) - 0.5

        #similarly, define the level 2 weights and biases
        self.hidden = np.zeros((12, 1))
        self.a2 = np.random.random((10, 12)) - 0.5
        self.b2 = np.random.random((10, 1)) - 0.5

        #finally, our output layer
        self.output = np.zeros((10, 1))
        
    def get(self, data):
        self.input = data

    def forward(self):
        #forward propogation is simple:
        #multiply out the input and the weights,
        #and sotre it in the next layer

        #then, add biases and apply the sigmoid function
        np.matmul(self.a1, self.input, self.hidden)
        self.hidden += self.b1
        self.hidden = sigmoid(self.hidden)

        np.matmul(self.a2, self.hidden, self.output)
        self.output += self.b2
        self.output = sigmoid(self.output)

    def getcost(self, data):
        #technically, this is error, not cost
        #regardless, it is the difference between
        #the true and target output
        self.costs = self.output - data

    def backward(self, learn):

        #sig is the derivative of the sigmoid function
        sig = self.output * (1 - self.output)
        #the bias change is the derivative of the sigmoid
        #times double the error (the derivative of the cost function
        #with respect to the bias)
        db2 = 2 * (np.multiply(self.costs,sig))
        #the weight change is this times the transform of
        #the hidden layer (this is the derivative
        #of the cost function with respect to the weights)
        da2 = np.dot(db2, self.hidden.T)


        #hidden costs can be found through the dot product
        #of the weight's transforms and the costs
        hidcosts = np.dot(self.a2.T, self.costs)

        #repeat to find layer 1 weight and bias changes
        sig = self.hidden * (1 - self.hidden)
        db1 = 2 * (np.multiply(hidcosts,sig))
        da1 = np.dot(db1, self.input.T)

        #finally, update the weights and biases
        #by the learn rate
        self.a2 -= da2 * learn
        self.b2 -= db2 * learn
        self.a1 -= da1 * learn
        self.b1 -= db1 * learn
