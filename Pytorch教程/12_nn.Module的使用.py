import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input):
        output = input + 1
        return output

model = Model()
input = torch.tensor([1.0])
output = model(input)
print(output)