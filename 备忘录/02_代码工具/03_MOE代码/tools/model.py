import torch
from torch import nn

class Expert(nn.Module):
    def __init__(self, embed_dims):
        super(Expert, self).__init__()
        self.conv3 = nn.Sequential(nn.Conv1d(embed_dims, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.Conv1d(64, 32, 1, 1, 0), nn.BatchNorm1d(32), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(embed_dims, 64, 5, 1, 2), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.Conv1d(64, 32, 1, 1, 0), nn.BatchNorm1d(32), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv1d(embed_dims, 64, 7, 1, 3), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.Conv1d(64, 32, 1, 1, 0), nn.BatchNorm1d(32), nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv1d(embed_dims, 64, 9, 1, 4), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.Conv1d(64, 32, 1, 1, 0), nn.BatchNorm1d(32), nn.ReLU())

        # transformer
        self.tf_encoder_layer = nn.TransformerEncoderLayer(128, nhead=8, batch_first=True)
        self.tf_encoder = nn.TransformerEncoder(self.tf_encoder_layer, num_layers=1)  # 一层和两层效果差不多
        self.dense = nn.Sequential(nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 2))

        # gru (比transformer稍差一点，考虑到运算速度，用transformer)
        # self.gru = nn.GRU(128, 64, num_layers=1, bidirectional=True)
        # self.dense = nn.Sequential(nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, X):
        X = X.permute(0, 2, 1)  # (N, C, L)
        X = torch.cat([self.conv3(X), self.conv5(X), self.conv7(X), self.conv9(X)], dim=1)
        X = X.permute(0, 2, 1)  # (N, L, C)

        # transformer
        X = self.tf_encoder(X)

        # gru
        # X, _ = self.gru(X)

        # Mean pooling
        X = torch.mean(X, dim=1)
        X = self.dense(X)
        return X

class Gate(nn.Module):
    def __init__(self, embed_dims, num_experts=4):
        super(Gate, self).__init__()
        self.num_experts = num_experts
        self.embed_dims = embed_dims
        self.dense = nn.Sequential(
            nn.Linear(self.embed_dims, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_experts),
            nn.Softmax(dim=1))

    def forward(self, X):
        X = X.mean(dim=1)
        X = self.dense(X)
        return X

class MOE(nn.Module):
    def __init__(self, trained_experts: list, embed_dims):
        super(MOE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        self.embed_dims = embed_dims
        self.gate = Gate(self.embed_dims, self.num_experts)

    def forward(self, X):
        weights = self.gate(X)
        # 后两维：每一列是一个专家的预测结果
        outputs = torch.stack([expert(X) for expert in self.experts], dim=2)
        weights = weights.unsqueeze(1).expand_as(outputs)
        return torch.sum(outputs * weights, dim=2)


if __name__ == '__main__':
    pass
