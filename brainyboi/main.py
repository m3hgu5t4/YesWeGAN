#import modules
from neural import Net
import numpy as np
import struct
from random import randint
import pickle

#this is the learning rate of the network,
# there are probably more effective values
learn = 0.1

#reads a single byte from a file in our data set
#and returns a 1-byte integer (ie a value between
#0 and 255
def read(file):
    return struct.unpack(">B", file.read(1))[0]

#loads the training/testing data.
file = open('train-images.idx3-ubyte', 'rb')
labels = open('train-labels.idx1-ubyte', 'rb')

#creates and initialises the neural network
#(see the neural.py file)
brainboi = Net()

#trains the network
while True:
    #this is the total cost (what the network will try
    # to minimise)
    totalcost = 0

    #we will start at a random place in the data and process the next 500
    # pieces of it.
    start = randint(1, 2000)
    end = start + 500
    for i in range(start, end):
        #the images each take up 784 bytes (28*28), plus a 16 byte header
        #and the labels each take up 1
        #so we jump to the appropriate place in the file
        file.seek((i*784) + 16,0)
        labels.seek(i + 8, 0)

        #read the image as a 28 by 28 matrix, then unravel it into a vector
        #this is not very efficient and could be changed to a single line
        data = np.matrix([[read(file)/255 for x in range(28)] for i in range(28)])
        data = np.reshape(data, (784, 1))

        #read the labels, then create a target vector with 0s in every position
        #except for the correct one, which is a one
        out = np.zeros((10, 1))
        real = read(labels)
        out[real] = 1

        #feed in our data
        brainboi.get(data)
        #forward propogate it
        brainboi.forward()
        #calculate costs and update the total costs
        brainboi.getcost(out)
        totalcost += brainboi.costs.sum()
        #backpropogate, using our learning rate defined earlier
        brainboi.backward(learn)

    #regulaly print the total cost of each batch
    #for monitoring purposes
    print(totalcost)

    #save our neural network periodically
    with open("brainyboi.nnw", "wb+") as save:
        pickle.dump(brainboi, save)
