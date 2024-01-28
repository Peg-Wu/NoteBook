# 使用说明见：Line 94 - 105 (Language: Chinese)

"""
OneHotEmbedder                     # 4
DNABert2Embedder                   # 768
DNABertEmbedder                    # 768
NucleotideTransformerEmbedder      # 1280
GENALMEmbedder                     # 768
GROVEREmbedder                     # 768
"""
import os
import torch
import pandas as pd
import numpy as np
from .bend.utils.embedders import OneHotEmbedder, DNABert2Embedder, DNABertEmbedder, NucleotideTransformerEmbedder, \
    GENALMEmbedder, GROVEREmbedder
import h5py
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='Running Device')
parser.add_argument('--h5_file', type=str, default='../embed.h5', help='The path to the h5 file')
parser.add_argument('--model_root', type=str, default='../pretrained_models', help='The root directory of the pretrained model')
parser.add_argument('--data_root', type=str, default='../data', help='The root directory of the data')
opt = parser.parse_args()
opt.device = torch.device(opt.device)

def embed_file_onehot(csv_file_path, save_path=None):
    # A: [1, 0, 0, 0]
    # C: [0, 1, 0, 0]
    # G: [0, 0, 1, 0]
    # T: [0, 0, 0, 1]
    data = pd.read_csv(csv_file_path)
    embedder = OneHotEmbedder(nucleotide_categories=['A', 'C', 'G', 'T'])
    result = embedder.embed(data.seq.values.tolist(), return_onehot=True, upsample_embeddings=True)
    X = torch.from_numpy(np.concatenate(result, axis=0)).float()
    y = torch.from_numpy(data.label.values).long()
    if save_path:
        torch.save([X, y], save_path)
    return X, y

def embed_file_dnabert2(csv_file_path, model_path, device, save_path=None):
    data = pd.read_csv(csv_file_path)
    embedder = DNABert2Embedder(model_path, device=device)
    result = embedder.embed(data.seq.values.tolist(), upsample_embeddings=True)
    X = torch.from_numpy(np.concatenate(result, axis=0)).float()
    y = torch.from_numpy(data.label.values).long()
    if save_path:
        torch.save([X, y], save_path)
    return X, y

def embed_file_dnabert(csv_file_path, model_path, kmer, device, save_path=None):
    data = pd.read_csv(csv_file_path)
    embedder = DNABertEmbedder(model_path, kmer=kmer, device=device)
    result = embedder.embed(data.seq.values.tolist(), upsample_embeddings=False)
    X = torch.from_numpy(np.concatenate(result, axis=0)).float()
    y = torch.from_numpy(data.label.values).long()
    if save_path:
        torch.save([X, y], save_path)
    return X, y

def embed_file_nucleotide(csv_file_path, model_path, device, save_path=None):
    data = pd.read_csv(csv_file_path)
    embedder = NucleotideTransformerEmbedder(model_path, device=device)
    result = embedder.embed(data.seq.values.tolist(), upsample_embeddings=True)
    X = torch.from_numpy(np.concatenate(result, axis=0)).float()
    y = torch.from_numpy(data.label.values).long()
    if save_path:
        torch.save([X, y], save_path)
    return X, y

def embed_file_genalm(csv_file_path, model_path, device, save_path=None):
    data = pd.read_csv(csv_file_path)
    embedder = GENALMEmbedder(model_path, device=device)
    result = embedder.embed(data.seq.values.tolist(), upsample_embeddings=True)
    X = torch.from_numpy(np.concatenate(result, axis=0)).float()
    y = torch.from_numpy(data.label.values).long()
    if save_path:
        torch.save([X, y], save_path)
    return X, y

def embed_file_grover(csv_file_path, model_path, device, save_path=None):
    data = pd.read_csv(csv_file_path)
    embedder = GROVEREmbedder(model_path, device=device)
    result = embedder.embed(data.seq.values.tolist(), upsample_embeddings=True)
    X = torch.from_numpy(np.concatenate(result, axis=0)).float()
    y = torch.from_numpy(data.label.values).long()
    if save_path:
        torch.save([X, y], save_path)
    return X, y

"""
以下部分用于将所有嵌入后的序列存到h5文件中，
- 如果你的数据量较大，又想要输出所有不同预训练模型的嵌入，不建议这样做，因为h5文件所占用的内存远大于预训练模型占用的内存
- 如果你的内存足够，这样做的好处是你不需要每次训练模型前都对序列进行嵌入，节省了大量的时间
- 如果你的数据量很小，存成h5文件后的占用内存小于预训练模型，这样做有利无弊
- 预训练模型pretrained_models.zip占5.95G，压缩前占6.7G，我们的训练和测试集一共有31952条序列，h5文件占13.82G
- 在我们的实验中，我们将其存成了h5文件

- 使用方式：
cd tools/
python seq_embedder.py --data_root='...' --model_root='...' --h5_file='...' --device='...'
"""

def init_h5(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    for path in ['OneHotEmbedder/train',
                 'OneHotEmbedder/test',
                 'DNABert2Embedder/train',
                 'DNABert2Embedder/test',
                 'DNABertEmbedder/3mer/train',
                 'DNABertEmbedder/3mer/test',
                 'DNABertEmbedder/4mer/train',
                 'DNABertEmbedder/4mer/test',
                 'DNABertEmbedder/5mer/train',
                 'DNABertEmbedder/5mer/test',
                 'DNABertEmbedder/6mer/train',
                 'DNABertEmbedder/6mer/test',
                 'NucleotideTransformerEmbedder/train',
                 'NucleotideTransformerEmbedder/test',
                 'GENALMEmbedder/bigbird/train',
                 'GENALMEmbedder/bigbird/test',
                 'GENALMEmbedder/bert/train',
                 'GENALMEmbedder/bert/test',
                 'GROVEREmbedder/train',
                 'GROVEREmbedder/test']:
        with h5py.File(file_path, 'a') as f:
            f.create_group(path)
    return None

def embed_csv_list(csv_list, h5_file_path, model_path_root, device, train=True, order=None):
    # h5_file : after initialization
    if order is None:
        order = ['A', 'C', 'G', 'U']

    if train:
        sub_dir = 'train'
    else:
        sub_dir = 'test'

    for i, csv in enumerate(csv_list):
        # onehot
        print(f'{sub_dir}_OneHotEmbedder_{order[i]}2OM:')
        X, _ = embed_file_onehot(csv)
        X = X.to(device)
        with h5py.File(h5_file_path, 'a') as f:
            loc = f[f'OneHotEmbedder/{sub_dir}']
            loc.create_dataset(name=f'{order[i]}_X', shape=X.shape, dtype='f4', data=X.cpu())
            # loc.create_dataset(name=f'{order[i]}_y', shape=y.shape, dtype='i1', data=y)

        # dnabert2
        print(f'{sub_dir}_DNABert2Embedder_{order[i]}2OM:')
        X, _ = embed_file_dnabert2(csv, model_path=os.path.join(model_path_root, 'DNABert2Embedder'), device=device)
        X = X.to(device)
        with h5py.File(h5_file_path, 'a') as f:
            loc = f[f'DNABert2Embedder/{sub_dir}']
            loc.create_dataset(name=f'{order[i]}_X', shape=X.shape, dtype='f4', data=X.cpu())
            # loc.create_dataset(name=f'{order[i]}_y', shape=y.shape, dtype='i1', data=y)

        # dnabert
        for k in range(3, 7):
            print(f'{sub_dir}_DNABertEmbedder({k}mer)_{order[i]}2OM:')
            X, _ = embed_file_dnabert(csv, model_path=os.path.join(model_path_root, f'DNABertEmbedder/{k}mer'),
                                      kmer=k, device=device)
            X = X.to(device)
            with h5py.File(h5_file_path, 'a') as f:
                loc = f[f'DNABertEmbedder/{k}mer/{sub_dir}']
                loc.create_dataset(name=f'{order[i]}_X', shape=X.shape, dtype='f4', data=X.cpu())
                # loc.create_dataset(name=f'{order[i]}_y', shape=y.shape, dtype='i1', data=y)

        # nucleotide_transformer
        print(f'{sub_dir}_NucleotideTransformerEmbedder_{order[i]}2OM:')
        X, _ = embed_file_nucleotide(csv, model_path=os.path.join(model_path_root, 'NucleotideTransformerEmbedder'),
                                     device=device)
        X = X.to(device)
        with h5py.File(h5_file_path, 'a') as f:
            loc = f[f'NucleotideTransformerEmbedder/{sub_dir}']
            loc.create_dataset(name=f'{order[i]}_X', shape=X.shape, dtype='f4', data=X.cpu())
            # loc.create_dataset(name=f'{order[i]}_y', shape=y.shape, dtype='i1', data=y)

        # genalm
        for k in ['bigbird', 'bert']:
            print(f'{sub_dir}_GENALMEmbedder({k})_{order[i]}2OM:')
            X, _ = embed_file_genalm(csv, model_path=os.path.join(model_path_root, f'GENALMEmbedder/{k}'),
                                     device=device)
            X = X.to(device)
            with h5py.File(h5_file_path, 'a') as f:
                loc = f[f'GENALMEmbedder/{k}/{sub_dir}']
                loc.create_dataset(name=f'{order[i]}_X', shape=X.shape, dtype='f4', data=X.cpu())
                # loc.create_dataset(name=f'{order[i]}_y', shape=y.shape, dtype='i1', data=y)

        # grover
        print(f'{sub_dir}_GROVEREmbedder_{order[i]}2OM:')
        X, _ = embed_file_grover(csv, model_path=os.path.join(model_path_root, 'GROVEREmbedder'),
                                 device=device)
        X = X.to(device)
        with h5py.File(h5_file_path, 'a') as f:
            loc = f[f'GROVEREmbedder/{sub_dir}']
            loc.create_dataset(name=f'{order[i]}_X', shape=X.shape, dtype='f4', data=X.cpu())
            # loc.create_dataset(name=f'{order[i]}_y', shape=y.shape, dtype='i1', data=y)

def main(data_root, h5_file_path, model_path_root, device):
    # init h5 file
    init_h5(h5_file_path)

    train_csv_path = os.path.join(data_root, '2OM_Train/csv')
    test_csv_path = os.path.join(data_root, '2OM_Test/csv')

    # train_csv, test_csv  |  order：[A, C, G, U]
    train_csv_list = sorted([os.path.join(train_csv_path, each) for each in os.listdir(train_csv_path)])
    test_csv_list = sorted([os.path.join(test_csv_path, each) for each in os.listdir(test_csv_path)])

    # train files process
    print('###################################  Train File Process  ###################################')
    embed_csv_list(train_csv_list, h5_file_path, model_path_root, device, train=True)

    # test files process
    print('###################################   Test File Process   ##################################')
    embed_csv_list(test_csv_list, h5_file_path, model_path_root, device, train=False)


if __name__ == '__main__':
    main(opt.data_root, opt.h5_file, opt.model_root, opt.device)
