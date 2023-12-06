import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

dataset = datasets.CIFAR10(root="CIFAR10", train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 2),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(32, 32, 5, 1, 2),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(32, 64, 5, 1, 2),
                                   nn.MaxPool2d(2),
                                   nn.Flatten(),
                                   nn.Linear(64*4*4, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 10))

    def forward(self, input):
        return self.model(input)

model = Model()
loss = nn.CrossEntropyLoss()

for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    result_loss = loss(output, targets)
    # Debug一下，看看梯度的变化
    result_loss.backward()
    break
