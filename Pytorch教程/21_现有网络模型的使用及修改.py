from torchvision import datasets, transforms, models
import torch.nn as nn

dataset = datasets.CIFAR10(root="./CIFAR10", train=False, transform=transforms.ToTensor(), download=True)

vgg = models.vgg16(pretrained=False)
# print(vgg)

# 在原vgg的基础上添加模块
vgg.add_module("added1", nn.Linear(100, 10))
# print(vgg)

# 在classifier内部添加模块
vgg.classifier.add_module("added2", nn.Linear(100, 10))
# print(vgg)

# 修改vgg的已有模块
vgg.classifier[6] = nn.Linear(4096, 10)
# print(vgg)

# 取子模块
model = vgg.get_submodule("features")
# print(model)