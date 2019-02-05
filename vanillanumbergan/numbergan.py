# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 19:26:09 2019

@author: m3hgu5t4
"""

# https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

# log the progress
from utils import Logger

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

# Load data
data = mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

# convert images and vectors into each other
def images_to_vectors(images):
    return images.view(images.size(0), 784)
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n

# creating the discriminator
class DiscriminatorNet(nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        
        n_features = 784 # number of inputs
        n_out = 1 # number of outputs
        
        # all Linear -> LeakyReLu -> Dropout
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024), # everyone's favourite wx+b
            nn.LeakyReLU(0.2), # if x < 0, output 0.2x
            nn.Dropout(0.3) # to "prevevnt overfitting"
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid() # get something between 0 and 1
        )
        
    def forward(self, x): # do the layers
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

discriminator = DiscriminatorNet()

# creating the generator
class GeneratorNet(nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        
        n_features = 100
        n_out = 784
        
        # the reverse of DiscriminatorNet
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256), # zoop up a size instead of down
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
generator = GeneratorNet()

# optimise this boi
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# is this loss function
loss = nn.BCELoss()

# targets: all 1s, and all 0s
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

# train the networks
def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake
def train_generator(optimizer, fake_data):
    N = fake_data.size(0)

    # Reset gradients
    optimizer.zero_grad()

    # Sample noise and generate fake data
    prediction = discriminator(fake_data)

    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()

    # Update weights with gradients
    optimizer.step()

    # Return error
    return error
# "static batch of noise" that will help show how the generator improves
num_test_samples = 16
test_noise = noise(num_test_samples)

# train the boi
def train(num_epochs):
    # Create logger instance
    logger = Logger(model_name='VGAN', data_name='MNIST')

    for epoch in range(num_epochs):
        for n_batch, (real_batch,_) in enumerate(data_loader):
            N = real_batch.size(0)

            # 1. Train Discriminator
            real_data = Variable(images_to_vectors(real_batch))

            # Generate fake data and detach 
            # (so gradients are not calculated for generator)
            fake_data = generator(noise(N)).detach()

            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

            # 2. Train Generator

            # Generate fake data
            fake_data = generator(noise(N))

            # Train G
            g_error = train_generator(g_optimizer, fake_data)

            # Log batch error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # Display Progress every few batches
            if (n_batch) % 300 == 0: 
                test_images = vectors_to_images(generator(test_noise))
                test_images = test_images.data

                logger.log_images(
                    test_images, num_test_samples, 
                    epoch, n_batch, num_batches
                );
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )

if __name__ == "__main__":
    # Total number of epochs to train
    num_epochs = 200
    train(num_epochs)
