import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import exists
from send_request import send_request

import pytest

@pytest.fixture
def dataset():
    DATA_PATH = 'data/cifar-10-python.tar.gz'
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    download = True

    # check if the data has already been loaded to avoid duplicate data loading
    if exists(DATA_PATH):
        download = False

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataset = {
        "batch_size": batch_size,
        "trainloader": trainloader,
        "testloader": testloader,
        "classes": classes
    }
    
    return dataset


@pytest.fixture
def net():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    return Net()

@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()

@pytest.fixture
def optimizer():
    return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def test_mnist_load():
    imports = "import tensorflow as tf"
    function_to_run = "tf.keras.datasets.mnist.load_data()"
    function_args = None
    function_kwargs = None

    result = send_request(imports, function_to_run, function_args, function_kwargs)

    # compare result from server with the real results
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert(result[0][0].all() == x_train.all())
    assert(result[0][1].all() == y_train.all())
    assert(result[1][0].all() == x_test.all())
    assert(result[1][1].all() == y_test.all())