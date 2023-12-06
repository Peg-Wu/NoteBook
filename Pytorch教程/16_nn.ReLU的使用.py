import torch
from torch import nn

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (1, 1, 2, 2))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input):
        output = self.relu(input)
        return output

model = Model()
output = model(input)
print(output)