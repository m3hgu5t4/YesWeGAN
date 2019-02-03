from neural import Net
import numpy as np
import struct
from random import randint
import pickle

#define our read function and load in the data
def read(file):
    return struct.unpack(">B", file.read(1))[0]

file = open('train-images.idx3-ubyte', 'rb')
labels = open('train-labels.idx1-ubyte', 'rb')

#load the neural network from a file
with open("brainyboi2.nnw", "rb") as save:
    brainboi = pickle.load(save)

#testing
for j in range(1):
    correct = 0
    failed = 0
    for n in range(2500, 3000):
        file.seek((n*784) + 16,0)
        labels.seek(n + 8, 0)

        data = np.matrix([[read(file)/255 for x in range(28)] for i in range(28)])
        data = np.reshape(data, (784, 1))
         
        brainboi.get(data)
        brainboi.forward()

        real = read(labels)
        guess = brainboi.output.argmax()

        if real == guess:
            correct += 1
        else:
            failed += 1

    print(correct/5)
