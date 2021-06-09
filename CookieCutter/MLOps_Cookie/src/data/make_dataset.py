import torch
import numpy
from torchvision import datasets, transforms

def mnist():
    # exchange with the real mnist dataset
 
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    trainset = datasets.MNIST('./data/', download=True, train=True, transform=transform)

    # Download and load the test data
    testset = datasets.MNIST('./data/', download=True, train=False, transform=transform)
    return trainset, testset

mnist()