import torch
import torchvision
import os

vgg16 = torchvision.models.vgg16(pretrained=False)

if not os.path.exists("./model"):
    os.makedirs("./model")

# 模型保存方式一：模型结构 + 模型参数
torch.save(vgg16, "./model/vgg16_method1.pth")
# 模型加载方式一
model = torch.load("./model/vgg16_method1.pth")

# 模型保存方式二：模型参数（官方推荐）
torch.save(vgg16.state_dict(), "./model/vgg16_method2.pth")
# 模型加载方式二
vgg16.load_state_dict(torch.load("./model/vgg16_method2.pth"))
