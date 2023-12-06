import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

trans = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="./CIFAR10",
                                         train=True,
                                         transform=trans,
                                         download=True)

test_set = torchvision.datasets.CIFAR10(root="./CIFAR10",
                                        train=False,
                                        transform=trans,
                                        download=True)

writer = SummaryWriter("logs")
for i in range(10):
    writer.add_image("test_set", test_set[i][0], i)
writer.close()

'''
print(test_set[0])
print(test_set.classes)

img, target = test_set[0]
# 图片：PIL.Image
print(img)
# 类别：数字表示
print(target)
# 类别：具体类别
print(test_set.classes[target])
# 查看图片
img.show()
'''