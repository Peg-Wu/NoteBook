from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

test_set = datasets.CIFAR10(root="./CIFAR10",
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

test_loader = DataLoader(dataset=test_set,
                         batch_size=64,
                         shuffle=True,  # 每轮epoch之后都会对所有数据重新洗牌
                         num_workers=0,
                         drop_last=False)

img, target = test_set[0]

print(img.shape)
print(target)

writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data  # 每次随机抓取4张图片，不是从头抓取！
        # print(imgs.shape)  # (B, C, H, W)
        # print(targets)  # (B)
        writer.add_images("epoch:{}".format(epoch), imgs, step)
        step += 1
writer.close()