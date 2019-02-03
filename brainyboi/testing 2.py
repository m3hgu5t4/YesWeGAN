from neural import Net
import numpy as np
import struct
from random import randint
import pickle
from PIL import Image

with open("brainyboi2.nnw", "rb") as save:
    brainboi = pickle.load(save)

for i in range(1000):
    input()

    data = Image.open("digit.png").convert('RGBA')
    data = np.array(data)
    data = data[:,:,1]
    data = np.reshape(data, (784, 1))
    data = np.divide(data, 255)
    
    
    brainboi.get(data)
    brainboi.forward()
    print(brainboi.output)
    guess = brainboi.output.argmax()
    print(guess)
