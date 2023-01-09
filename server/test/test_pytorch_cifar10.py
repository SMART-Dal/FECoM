import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import exists
# add parent directory to path
import sys
sys.path.insert(0,'..')
from send_request import send_request

import pytest


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
    return Net()

@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()

@pytest.fixture
def training_batch(dataset):
    """
    returns tuple(inputs, labels) for one batch from the training dataset
    """
    trainloader = dataset["trainloader"]
    return next(iter(trainloader))


def test_optimizer_zero_grad(net):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    imports = "import torch.optim as optim"
    function_to_run = "obj.zero_grad()"
    method_object = optimizer

    return_dict = send_request(imports, function_to_run, method_object=method_object)
    zero_grad_optimizer = return_dict["method_object"]

    optimizer.zero_grad()

    assert return_dict["return"] is None
    assert zero_grad_optimizer.state_dict() == optimizer.state_dict()


def test_forward_pass(net, training_batch):
    inputs = training_batch[0]

    imports = """import torch
import torch.nn as nn
import torch.nn.functional as F"""
    custom_class =  """class Net(nn.Module):
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
        return x"""
    function_to_run = "obj(*args)"
    method_object = net
    function_args = [inputs]

    return_dict = send_request(imports, function_to_run, function_args=function_args, method_object=method_object, custom_class=custom_class)

    outputs = net(inputs)

    assert torch.equal(return_dict["return"], outputs)

def test_loss_function(net, training_batch, criterion):
    labels = training_batch[1]
    outputs = net(training_batch[0])

    imports = "import torch.nn as nn"
    function_to_run = "obj(*args)"
    method_object = criterion
    function_args = [outputs, labels]

    return_dict = send_request(imports, function_to_run, function_args, method_object=method_object)

    assert return_dict["return"] == criterion(outputs, labels)

# incomplete
def test_backward_pass(net, training_batch, criterion):
    labels = training_batch[1]
    outputs = net(training_batch[0])
    loss = criterion(outputs, labels)

    imports = "import torch"
    function_to_run = "obj.backward()"
    method_object = loss
    required_data = [outputs, ]

    return_dict = send_request(imports, function_to_run, method_object=method_object)

    assert torch.equal(return_dict["method_object"], loss)
    # TODO also test whether the backward pass correctly stores gradients in the associated tensors (outputs, net parameters).
    # This will likely require an extension to the current API such that these tensors can be sent to the server, too
    # see this for more on the internal functioning of torch
    # https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step

# incomplete, see above 
def test_optimizer_step(net, training_batch, criterion):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    labels = training_batch[1]
    outputs = net(training_batch[0])
    loss = criterion(outputs, labels)
    loss.backward()

    imports = "import torch.optim as optim"
    function_to_run = "obj.step()"
    method_object = optimizer
    print(optimizer)

    return_dict = send_request(imports, function_to_run, method_object=method_object)
    # TODO add assertions, see above