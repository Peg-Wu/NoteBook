# 预测人类RNA中2'-O-甲基化(2OM)位点

- 基于Mixture Of Experts(MOE)模型
  - Expert专家模型仅用了两层CNN和全连接层，**模型待完善中...**
  - Gate门控模型仅用了两层全连接层，**模型待完善中...**
  - 该模型包含4个Experts，每个Expert模型架构目前是一致的，**这4个专家分别处理A2OM、G2OM、C2OM和U2OM数据**
- 训练请运行`Train.py`
- 测试请运行`Test.py`
- 本项目提供了MOE模型训练可视化，在训练过程中，会生成`./logs`文件，请使用`Tensorboard`查看！

## 🎨训练策略说明

1. 专家模型在训练X个epoch后，如果`训练损失`没有降低，则提前停止训练
2. MOE模型在训练X个epoch后，如果`验证损失`没有降低，则提前停止训练
3. 第一阶段：训练专家模型，`专家模型没有验证集`，只进行训练，根据训练损失随时终止
4. 第二阶段：训练MOE模型，`MOE模型有验证集`，并记录了Accuracy、MCC、Precision、F1、Sp、Sn等指标

## 测试集泛化情况

- Model1: Expert模型基于**CNN-Transformer**

```python
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
        self.conv3 = nn.Sequential(nn.Conv1d(4, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                   conv1d_k1_block(64, 32))
        self.conv5 = nn.Sequential(nn.Conv1d(4, 64, 5, 1, 2), nn.BatchNorm1d(64), nn.ReLU(),
                                   conv1d_k1_block(64, 32))
        self.conv7 = nn.Sequential(nn.Conv1d(4, 64, 7, 1, 3), nn.BatchNorm1d(64), nn.ReLU(),
                                   conv1d_k1_block(64, 32))
        self.conv9 = nn.Sequential(nn.Conv1d(4, 64, 9, 1, 4), nn.BatchNorm1d(64), nn.ReLU(),
                                   conv1d_k1_block(64, 32))
        self.tf_encoder_layer = nn.TransformerEncoderLayer(128, 8, batch_first=True)
        self.tf_encoder = nn.TransformerEncoder(self.tf_encoder_layer, 3)
        self.dense = nn.Sequential(nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, X):
        X = X.permute(0, 2, 1)  # (batch, 4, 41)
        # 将one_hot作为特征维进行卷积
        X = torch.cat([self.conv3(X), self.conv5(X), self.conv7(X), self.conv9(X)], dim=1)  # (batch, 128, 41)
        X = X.permute(0, 2, 1)  # (batch_size, 41, 128)
        X = self.tf_encoder(X)
        # Mean pooling
        X = torch.mean(X, dim=1)
        X = self.dense(X)
        return X

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
```

> MOE:
> Accuracy: 0.8095628413088627
> Precision: 0.8114348538694696
> F1: 0.8089887638232424
> MCC: 0.6191368675842226
> Sn: 0.8065573766084385
> Sp: 0.8125683055669025

