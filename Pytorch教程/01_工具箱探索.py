import torch

# dir()函数，能让我们知道工具箱以及工具箱中的分隔区有什么东西
dir(torch)
dir(torch.cuda)

# help()函数，能让我们知道每个工具是如何使用的，工具的使用方法
help(torch.cuda.is_available)