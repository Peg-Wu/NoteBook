import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import os
import warnings
warnings.filterwarnings("ignore")

def convert_string_to_numbers(input_str):
    mapping = {'A': 0, 'G': 1, 'C': 2, 'U': 3}
    return [mapping[char] for char in input_str]

def convert_sequences_to_numbers(str_array):
    return np.array([convert_string_to_numbers(each) for each in str_array.tolist()])

def get_dataset(x_array, y_array, Dataset_type, *args):
    """Dataset_type为自定义的Dataset类, 以及需要额外传入的属性"""
    X = torch.from_numpy(x_array).float()
    y = torch.from_numpy(y_array).long()
    return Dataset_type(X, y, *args)

def split_dataset(my_dataset, ratio):
    num1 = int(len(my_dataset) * ratio)
    num2 = len(my_dataset) - num1
    part1, part2 = random_split(dataset=my_dataset,
                                lengths=[num1, num2])
    return part1, part2

def get_dataloader(my_dataset, batch_size, train=True, pin_memory=False):
    if train:
        return DataLoader(my_dataset, batch_size, shuffle=True, pin_memory=pin_memory)
    else:
        return DataLoader(my_dataset, batch_size, shuffle=False, pin_memory=pin_memory)

class PositionalEncoding(nn.Module):
    """位置编码"""
    # num_hiddens表示嵌入维度
    def __init__(self, num_hiddens, dropout, max_len):
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

class Dataset_onehot(Dataset):
    def __init__(self, X, y):
        super(Dataset_onehot, self).__init__()
        self.X = X
        self.y = y
        self.onehot_dim = 4
        self.max_len = self.X.shape[-1]
        self.pe = PositionalEncoding(self.onehot_dim, dropout=0.1, max_len=self.max_len)

    def __getitem__(self, index):
        X = F.one_hot(self.X.long(), num_classes=self.onehot_dim)
        X = self.pe(X.float())
        y = self.y.long()
        return X[index], y[index]

    def __len__(self):
        return len(self.y)

class Dataset_word2vec(Dataset):
    def __init__(self, X, y, embed_dim):
        super(Dataset_word2vec, self).__init__()
        self.X = X
        self.y = y
        self.max_len = self.X.shape[-1]
        self.embed_dim = embed_dim
        self.word2vec = nn.Embedding(4, self.embed_dim)
        self.pe = PositionalEncoding(self.embed_dim, dropout=0.1, max_len=self.max_len)

    def __getitem__(self, index):
        X = self.word2vec(self.X.long())
        X = self.pe(X.float())
        y = self.y.long()
        return X[index], y[index]

    def __len__(self):
        return len(self.y)

def get_train_dataset_dataloader(A_path, G_path, C_path, U_path,
                                 ratio, batch_size, Dataset_type, *args, pin_memory=False):
    """
    从csv文件构建用于训练expert和moe的dataset和dataloader
    :param A_path: A2OM_train.csv文件路径
    :param G_path: G2OM_train.csv文件路径
    :param C_path: C2OM_train.csv文件路径
    :param U_path: U2OM_train.csv文件路径
    :param ratio: 训练集比例
    :param batch_size: 批量大小
    :param Dataset_type: 如: Dataset_onehot
    :param args: 实例化Dataset_type需要的参数(除X和y外)
    :return: expert_dataset(AGCU 4个 type: list), expert_dataloader,
             moe_dataset(train & valid 2个 type: list), moe_dataloader
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
    A_dataset = get_dataset(A_seq, A_label, Dataset_type, *args)
    G_dataset = get_dataset(G_seq, G_label, Dataset_type, *args)
    C_dataset = get_dataset(C_seq, C_label, Dataset_type, *args)
    U_dataset = get_dataset(U_seq, U_label, Dataset_type, *args)

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
    moe_train_dataloader = get_dataloader(moe_train_dataset, batch_size, train=True, pin_memory=pin_memory)
    moe_valid_dataloader = get_dataloader(moe_valid_dataset, batch_size, train=False, pin_memory=pin_memory)
    moe_dataloader = [moe_train_dataloader, moe_valid_dataloader]
    expert_dataloader = [get_dataloader(each, batch_size, train=True, pin_memory=pin_memory) for each in expert_dataset]

    return expert_dataset, expert_dataloader, moe_dataset, moe_dataloader


def get_test_dataset_dataloader(test_data_path, batch_size, Dataset_type, *args, integrate=True, pin_memory=False):
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
        dataset = get_dataset(convert_sequences_to_numbers(all_data.seq.values), all_data.label.values,
                              Dataset_type, *args)
        dataloader = get_dataloader(dataset, train=False, batch_size=batch_size, pin_memory=pin_memory)
        return dataset, dataloader
    else:
        dataset = [get_dataset(convert_sequences_to_numbers(each.seq.values), each.label.values,
                               Dataset_type, *args) for each in [A, G, C, U]]
        dataloader = [get_dataloader(each, train=False, batch_size=batch_size, pin_memory=pin_memory) for each in dataset]
        return dataset, dataloader

if __name__ == '__main__':
    pass
