import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import os

def convert_string_to_numbers(input_str):
    mapping = {'A': 0, 'G': 1, 'C': 2, 'U': 3}
    return [mapping[char] for char in input_str]

def convert_sequences_to_numbers(str_array):
    return np.array([convert_string_to_numbers(each) for each in str_array.tolist()])

class MyDataset(Dataset):
    def __init__(self, X, y):
        super(MyDataset, self).__init__()
        self.X = X
        self.y = y
        self.onehot_dim = 4
        self.embed_dim = self.onehot_dim
        self.max_len = self.X.shape[-1]
        self.dropout = 0.1
        self.pe = PositionalEncoding(self.embed_dim, dropout=self.dropout, max_len=self.max_len)

    def __getitem__(self, index):
        X = F.one_hot(self.X.long(), num_classes=self.onehot_dim)
        X = self.pe(X.float())
        y = self.y.long()
        return X[index], y[index]

    def __len__(self):
        return len(self.y)

def get_dataset(x_array, y_array):
    X = torch.from_numpy(x_array).float()
    y = torch.from_numpy(y_array).long()
    return MyDataset(X, y)

def split_dataset(my_dataset, ratio=0.5, seed=520):
    for_expert = int(len(my_dataset) * ratio)
    for_moe = len(my_dataset) - for_expert
    torch.manual_seed(seed)
    part1, part2 = random_split(dataset=my_dataset,
                                lengths=[for_expert, for_moe])
    return part1, part2

def get_dataloader(my_dataset, batch_size=128, train=True):
    if train:
        return DataLoader(my_dataset, batch_size, shuffle=True)
    else:
        return DataLoader(my_dataset, batch_size*2, shuffle=False)

class PositionalEncoding(nn.Module):
    """位置编码"""
    # num_hiddens表示嵌入维度
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        # batch_size是1，每一行是一个被嵌入后的token
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)  # 偶数列用sin
        self.P[:, :, 1::2] = torch.cos(X)  # 奇数列用cos

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        # 加Dropout是为了防止模型对position太过敏感
        return self.dropout(X)

def get_test(test_data_path=r'./data/2OM_Test/csv', integrate=True):
    """
    该函数用于得到测试数据集，dataloader一次性将所有的X和y加载出来
    :param test_data_path: 测试数据集路径
    :param integrate: 是否整合所有的测试数据集
    :return: dataset和dataloader，如果integrate=False，则返回的dataset和dataloader为列表，分别对应AGCU的dataset和dataloader
    """
    A = pd.read_csv(os.path.join(test_data_path, 'A2OM_test.csv'))
    G = pd.read_csv(os.path.join(test_data_path, 'G2OM_test.csv'))
    C = pd.read_csv(os.path.join(test_data_path, 'C2OM_test.csv'))
    U = pd.read_csv(os.path.join(test_data_path, 'U2OM_test.csv'))
    if integrate:
        all_data = pd.concat([A, G, C, U], axis=0)
        dataset = get_dataset(convert_sequences_to_numbers(all_data.seq.values), all_data.label.values)
        dataloader = get_dataloader(dataset, train=False, batch_size=len(dataset))
        return dataset, dataloader
    else:
        dataset = [get_dataset(convert_sequences_to_numbers(each.seq.values), each.label.values) for each in [A, G, C, U]]
        dataloader = [get_dataloader(each, train=False, batch_size=len(each)) for each in dataset]
        return dataset, dataloader

def main(A_path='./data/2OM_Train/csv/A2OM_train.csv',
         G_path='./data/2OM_Train/csv/G2OM_train.csv',
         C_path='./data/2OM_Train/csv/C2OM_train.csv',
         U_path='./data/2OM_Train/csv/U2OM_train.csv', ratio=0.7):
    """
    主程序：从csv文件构建用于训练expert和moe的dataset和dataloader
    :param A_path: A甲基化csv文件路径
    :param G_path: G甲基化csv文件路径
    :param C_path: C甲基化csv文件路径
    :param U_path: U甲基化csv文件路径
    :param ratio: 每个dataset切分给expert的比例
    :param moe_for_train: 对moe的dataset进行切分，训练集的比例
    :return: dataset & dataloader
    """
    # READ_FILES
    A, G, C, U = pd.read_csv(A_path), pd.read_csv(G_path), pd.read_csv(C_path), pd.read_csv(U_path)

    # SEQUENCE & LABEL
    A_seq, G_seq, C_seq, U_seq = A.seq.values, G.seq.values, C.seq.values, U.seq.values
    A_label, G_label, C_label, U_label = A.label.values, G.label.values, C.label.values, U.label.values

    # CONVERT SEQUENCE TO NUMBER
    A_seq = convert_sequences_to_numbers(A_seq)
    G_seq = convert_sequences_to_numbers(G_seq)
    C_seq = convert_sequences_to_numbers(C_seq)
    U_seq = convert_sequences_to_numbers(U_seq)

    # CONSTRUCT DATASET
    A_dataset = get_dataset(A_seq, A_label)
    G_dataset = get_dataset(G_seq, G_label)
    C_dataset = get_dataset(C_seq, C_label)
    U_dataset = get_dataset(U_seq, U_label)

    # SPLIT DATASET, dataset2 for validation, dataset1 for every expert
    A_dataset1, A_dataset2 = split_dataset(A_dataset, ratio)
    G_dataset1, G_dataset2 = split_dataset(G_dataset, ratio)
    C_dataset1, C_dataset2 = split_dataset(C_dataset, ratio)
    U_dataset1, U_dataset2 = split_dataset(U_dataset, ratio)
    valid_dataset = A_dataset2 + G_dataset2 + C_dataset2 + U_dataset2  # moe 验证集

    # EXPERT_DATASET & MOE_DATASET
    expert_dataset = [A_dataset1, G_dataset1, C_dataset1, U_dataset1]
    moe_train_dataset = A_dataset1 + G_dataset1 + C_dataset1 + U_dataset1
    moe_valid_dataset = valid_dataset
    moe_dataset = [moe_train_dataset, moe_valid_dataset]

    # EXPERT_DATALOADER & MOE_DATALOADER
    moe_train_dataloader = get_dataloader(moe_train_dataset)
    moe_valid_dataloader = get_dataloader(moe_valid_dataset, train=False)
    moe_dataloader = [moe_train_dataloader, moe_valid_dataloader]
    expert_dataloader = [get_dataloader(each) for each in expert_dataset]

    return expert_dataset, expert_dataloader, moe_dataset, moe_dataloader


if __name__ == '__main__':
    expert_dataset, expert_dataloader, moe_dataset, moe_dataloader = main(
        A_path='../data/2OM_Train/csv/A2OM_train.csv',
        G_path='../data/2OM_Train/csv/G2OM_train.csv',
        C_path='../data/2OM_Train/csv/C2OM_train.csv',
        U_path='../data/2OM_Train/csv/U2OM_train.csv',
        ratio=0.7)
