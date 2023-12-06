# 一般训练模块和测试模块分开：train.py和test.py

from PIL import Image
from torchvision import transforms
import torch
from torch import nn


img_path = "./dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

trans = transforms.Compose([transforms.Resize((32, 32)),
                            transforms.ToTensor()])

img = trans(img)
img = torch.reshape(img, (1, 3, 32, 32))  # 转换成（B，C，H，W）

# 网络模型
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

    def forward(self, x):
        x = self.model(x)
        return x

# 注意：必须要先把模型架构定义出来，才能调用torch.load()
model = torch.load("./model/model_1.pth", map_location=torch.device("cpu"))

# 开始测试
model.eval()
with torch.no_grad():
    output = model(img)

print(output.argmax(1))