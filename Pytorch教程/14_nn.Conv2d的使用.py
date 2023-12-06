import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = datasets.CIFAR10(root="./CIFAR10",
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, input):
        return self.conv2d(input)

model = Model()

writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs, targets = data
    output = model(imgs)

    print(imgs.shape)
    print(output.shape)

    # 写入tensorboard的图像通道数要求是3，我们把batch_size扩大
    output = torch.reshape(output, (-1, 3, output.shape[-1], output.shape[-1]))

    writer.add_images("pre", imgs, step)
    writer.add_images("after", output, step)
    step += 1

writer.close()