import torch
from torch import nn

# 每次经过卷积块，形状会减半，通道数由给定参数决定
def conv_block(num_convs, in_channels, out_channels, kernel_size):
    layers = []
    for _ in range(num_convs):
        # 保持输入和输出的形状相同
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=int((kernel_size-1)/2)))
        layers.append(nn.ReLU())
        in_channels = out_channels
    # 通过Pooling层将形状减半
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def conv_net(conv_arch, in_channels, kernel_size):
    conv_blks = []
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(conv_block(num_convs, in_channels, out_channels, kernel_size))
        in_channels = out_channels
    return nn.Sequential(*conv_blks)

class Expert(nn.Module):
    def __init__(self, embed_dims):
        super(Expert, self).__init__()
        conv_arch = tuple([(1, 128) for _ in range(5)])
        self.conv_k3 = conv_net(conv_arch, embed_dims, kernel_size=3)
        self.conv_dense = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, 8))
        self.kmer4_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 64), nn.ReLU())
        self.kmer3_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, 16), nn.ReLU())
        self.kmer2_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(32, 4), nn.ReLU())
        self.output_layer = nn.Linear(16, 2)


    def forward(self, X, *args):
        if len(args) == 1:
            X_features,  = args

        # Convert Dimension
        X = X.permute(0, 2, 1)  # (N, C, L)
        # CNN
        X = self.conv_k3(X)
        # Convert Dimension
        X = X.permute(0, 2, 1)  # (N, L, C)
        # Mean pooling
        X = torch.mean(X, dim=1)
        # Conv Dense
        X = self.conv_dense(X)
        # Features Model
        kmer_1 = X_features[:, :4]
        kmer_2 = X_features[:, 4:4+16]
        kmer_3 = X_features[:, 20:20+64]
        kmer_4 = X_features[:, 84:84+256]
        X_out = self.kmer4_fc(kmer_4)
        X_out = self.kmer3_fc(torch.cat([X_out, kmer_3], dim=-1))
        X_out = self.kmer2_fc(torch.cat([X_out, kmer_2], dim=-1))
        X_out = torch.cat([X_out, kmer_1], dim=-1)
        X = torch.cat([X, X_out], dim=-1)
        # Output
        X = self.output_layer(X)
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

    def forward(self, X):  # 门控网络只看预训练模型嵌入的特征
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

    def forward(self, X, *args):
        weights = self.gate(X)
        # 后两维：每一列是一个专家的预测结果
        outputs = torch.stack([expert(X, *args) for expert in self.experts], dim=2)
        weights = weights.unsqueeze(1).expand_as(outputs)
        return torch.sum(outputs * weights, dim=2)


if __name__ == '__main__':
    pass
