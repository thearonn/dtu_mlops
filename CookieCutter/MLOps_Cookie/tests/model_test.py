from make_dataset import mnist
from model import MyAwesomeModel

import torch


def test_model():
    model = MyAwesomeModel()
    train_set, test_set = mnist()

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True)

    for images, labels in trainloader:
        if images.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        assert (images.shape) == torch.Size([64, 1, 28, 28])
        assert labels.shape == torch.Size([64])
        output = model(images)
        # print(output.shape)
        assert (output.shape) == torch.Size([64, 10])
        break


test_model()
