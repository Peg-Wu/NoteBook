import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from script import model
import os
import numpy as np

def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
same_seed(520)

# 创建model文件夹，用于保存模型
if not os.path.exists("./model"):
    os.mkdir("./model")

# 准备数据集
train_dataset = datasets.CIFAR10(root="./CIFAR10", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root="./CIFAR10", train=False, transform=transforms.ToTensor(), download=True)

# 查看数据集的大小
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print(f"训练数据集的大小：{train_dataset_size}张")
print(f"测试数据集的大小：{test_dataset_size}张")

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=64*2, shuffle=False, num_workers=0)

# 创建网络模型
my_model = model.Model()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs")

for i in range(epoch):
    print(f"----------第{i+1}轮训练开始----------")

    # 训练步骤开始
    my_model.train()  # 对Dropout和BatchNorm等有作用
    for data in train_dataloader:
        imgs, targets = data
        output = my_model(imgs)
        loss = loss_fn(output, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step  % 100 == 0:  # 训练每100次打印一次损失
            print(f"训练次数：{total_train_step}, Loss：{loss.item()}")  # 注意：使用loss.item()！
            writer.add_scalar("train_loss", loss.item(), total_train_step)  # 每训练100次写入tensorboard中！

    # 测试步骤开始
    my_model.eval()  # 对Dropout和BatchNorm等有作用
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            output = my_model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            total_test_accuracy += (output.argmax(1) == targets).sum()

    print(f"整体测试集上的Loss：{total_test_loss}")
    print(f"整体测试集上的准确率：{total_test_accuracy / test_dataset_size}")
    total_test_step += 1
    writer.add_scalar("test_loss", total_test_loss, total_test_step)  # 其实也可以直接使用(epoch + 1)
    writer.add_scalar("test_accuracy", total_test_accuracy / test_dataset_size, total_test_step)

    torch.save(my_model, f"./model/model_{i + 1}.pth")
    # torch.save(my_model.state_dict(), f"./model/model_{i + 1}.pth")
    print("模型已保存！")

writer.close()
