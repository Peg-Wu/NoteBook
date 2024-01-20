import torch
from torch import nn

class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv1d(41, 64, 3, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv1d(41, 64, 5, 2, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.conv7 = nn.Sequential(
            nn.Conv1d(41, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.conv9 = nn.Sequential(
            nn.Conv1d(41, 64, 9, 2, 4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.dense = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2))

    def forward(self, X):
        X = torch.cat([self.conv3(X), self.conv5(X), self.conv7(X), self.conv9(X)], dim=1)
        # Mean pooling
        X = torch.mean(X, dim=-1)
        return self.dense(X)

class Gate(nn.Module):
    def __init__(self, num_experts=4):
        super(Gate, self).__init__()
        self.num_experts = num_experts
        self.dense = nn.Sequential(
            nn.Flatten(),
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

