import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

test_dataset = datasets.CIFAR10(root="./CIFAR10", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(64*3*32*32, 10)

    def forward(self, input):
        return self.linear(input)

model = Model()

# 使用torch.reshape()
for data in test_loader:
    imgs, targets = data
    imgs = torch.reshape(imgs, (1, 1, 1, -1))
    print(imgs.shape)
    output = model(imgs)
    print(output.shape)
    break

# 使用torch.flatten()
for data in test_loader:
    imgs, targets = data
    imgs = torch.flatten(imgs)
    print(imgs.shape)
    output = model(imgs)
    print(output.shape)
    break