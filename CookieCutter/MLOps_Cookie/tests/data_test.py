from make_dataset import mnist
from torch.utils.data.dataset import T
import torch

def data_test():
    train_set, test_set = mnist()


    assert len(train_set) == 60000
    assert len(test_set) == 10000
    assert next(iter(train_set))[0].shape == torch.Size([1, 28, 28])
    if next(iter(train_set))[0].shape != torch.Size([1, 28, 28]):
            raise ValueError('Input has wrong dimensions')
    assert next(iter(test_set))[0].shape == torch.Size([1, 28, 28])
    assert len(torch.unique(train_set.targets)) == 10
    #print(type(train_set))

data_test()