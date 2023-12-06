from torch import nn
import torch
import warnings
warnings.filterwarnings("ignore")

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 池化层默认的stride = kernel_size, 并且池化前后channel数不会变化！
        self.pool = nn.MaxPool2d(3, ceil_mode=True)

    def forward(self, input):
        return self.pool(input)

model = Model()
output = model(input)

print("池化前：{}".format(input.shape))
print("池化后：{}".format(output.shape))