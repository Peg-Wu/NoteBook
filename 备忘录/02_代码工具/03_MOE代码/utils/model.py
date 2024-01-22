import torch
from torch import nn

def conv1d_k1_block(in_channels, out_channels, padding=0, stride=1):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride, padding),
                         nn.BatchNorm1d(out_channels), nn.ReLU())

class CancelOut(nn.Module):
    '''
    CancelOut Layer
    x - an input data (vector, matrix, tensor)
    '''
    def __init__(self, inp, *kargs, **kwargs):
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(inp, requires_grad=True) + 4)

    def forward(self, x):
        return (x * torch.sigmoid(self.weights.float()))

class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()
        self.conv3 = nn.Sequential(nn.Conv1d(41, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                   conv1d_k1_block(64, 32))
        self.conv5 = nn.Sequential(nn.Conv1d(41, 64, 5, 1, 2), nn.BatchNorm1d(64), nn.ReLU(),
                                   conv1d_k1_block(64, 32))
        self.conv7 = nn.Sequential(nn.Conv1d(41, 64, 7, 1, 3), nn.BatchNorm1d(64), nn.ReLU(),
                                   conv1d_k1_block(64, 32))
        self.conv9 = nn.Sequential(nn.Conv1d(41, 64, 9, 1, 4), nn.BatchNorm1d(64), nn.ReLU(),
                                   conv1d_k1_block(64, 32))
        self.gru = nn.GRU(input_size=4, hidden_size=16, num_layers=2, batch_first=True)
        self.dense = nn.Sequential(nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, X):
        X = torch.cat([self.conv3(X), self.conv5(X), self.conv7(X), self.conv9(X)], dim=1)  # (batch, 128, 4)
        X, _ = self.gru(X)  # (batch, 128, hidden_size)
        # Mean pooling
        X = torch.mean(X, dim=-1)
        return self.dense(X)

class Gate(nn.Module):
    def __init__(self, num_experts=4):
        super(Gate, self).__init__()
        self.num_experts = num_experts
        self.dense = nn.Sequential(
            nn.Flatten(),
            CancelOut(41*4),
            nn.Linear(41*4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 4),
            nn.Softmax(dim=1))

    def forward(self, X):
        return self.dense(X)

class MOE(nn.Module):
    def __init__(self, trained_experts: list):
        super(MOE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        self.gate = Gate(num_experts=4)

    def forward(self, X):
        weights = self.gate(X)
        # 后两维：每一列是一个专家的预测结果
        outputs = torch.stack([expert(X) for expert in self.experts], dim=2)
        weights = weights.unsqueeze(1).expand_as(outputs)
        return torch.sum(outputs * weights, dim=2)
