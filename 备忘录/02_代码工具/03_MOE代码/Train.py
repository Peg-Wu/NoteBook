import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
from tools import data, train_script as ts, model
import argparse

# Hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=520)
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--h5_file', type=str, default='./embed.h5')
parser.add_argument('--train_ratio', type=float, default=0.7, help='train_dataset_ratio')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--early_stop', type=int, default=10)
parser.add_argument('--embed_type', default=['DNABertEmbedder/3mer'])
"""
--embed_type:
'OneHotEmbedder','DNABert2Embedder','DNABertEmbedder/3mer','DNABertEmbedder/4mer','DNABertEmbedder/5mer',
'DNABertEmbedder/6mer','NucleotideTransformerEmbedder','GENALMEmbedder/bigbird','GENALMEmbedder/bert','GROVEREmbedder'
"""

# TODO: Add Model Parameters
pass

parser.add_argument('--a_logs', type=str, default='./train_logs/a/db3')
parser.add_argument('--c_logs', type=str, default='./train_logs/c/db3')
parser.add_argument('--g_logs', type=str, default='./train_logs/g/db3')
parser.add_argument('--u_logs', type=str, default='./train_logs/u/db3')
parser.add_argument('--moe_logs', type=str, default='./train_logs/moe/db3')

parser.add_argument('--save_a', type=str, default='./model_param/a/db3/a.pkl')
parser.add_argument('--save_c', type=str, default='./model_param/c/db3/c.pkl')
parser.add_argument('--save_g', type=str, default='./model_param/g/db3/g.pkl')
parser.add_argument('--save_u', type=str, default='./model_param/u/db3/u.pkl')
parser.add_argument('--save_moe', type=str, default='./model_param/moe/db3/moe.pkl')
opt = parser.parse_args()

# Device
opt.device = torch.device(opt.device)

# Creat Directory
for each in [opt.a_logs, opt.c_logs, opt.g_logs, opt.u_logs, opt.moe_logs]:
    os.makedirs(each, exist_ok=True)
for each in [opt.save_a, opt.save_c, opt.save_g, opt.save_u, opt.save_moe]:
    os.makedirs(os.path.split(each)[0], exist_ok=True)

# Seed
ts.same_seed(opt.seed)

def generate_datasets(datasets):
    datasets_split = []
    for dataset in datasets:
        datasets_split.append(data.split_dataset_2parts(dataset, opt.train_ratio))
    train_dataset_moe, valid_dataset = \
        data.merge_datasets([each[0] for each in datasets_split]), \
        data.merge_datasets([each[1] for each in datasets_split])
    train_dataset_a, train_dataset_c, train_dataset_g, train_dataset_u = \
        [each[0] for each in datasets_split]
    return train_dataset_a, train_dataset_c, train_dataset_g, train_dataset_u, train_dataset_moe, valid_dataset

def generate_dataloaders(train_dataset_a, train_dataset_c, train_dataset_g, train_dataset_u, train_dataset_moe, valid_dataset):
    train_dataloader_a = DataLoader(train_dataset_a, batch_size=opt.batch_size, shuffle=True)
    train_dataloader_c = DataLoader(train_dataset_c, batch_size=opt.batch_size,  shuffle=True)
    train_dataloader_g = DataLoader(train_dataset_g, batch_size=opt.batch_size, shuffle=True)
    train_dataloader_u = DataLoader(train_dataset_u, batch_size=opt.batch_size, shuffle=True)
    train_dataloader_moe = DataLoader(train_dataset_moe, batch_size=opt.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    return train_dataloader_a, train_dataloader_c, train_dataloader_g, train_dataloader_u, train_dataloader_moe, valid_dataloader

# Dataset & DataLoader (Embedding), 如果使用多模态模型，则无需以下代码
# datasets = data.get_datasets(opt.data_root, opt.h5_file, opt.embed_type, bases='ACGU', train=True)
# train_dataloader_a, train_dataloader_c, train_dataloader_g, train_dataloader_u, train_dataloader_moe, valid_dataloader = \
#     generate_dataloaders(*generate_datasets(datasets))

# Dataset & DataLoader (Kmer Features), 注意：带有手动特征的Dataset包含了预训练模型编码的特征
kmers_datasets = data.get_datasets_kmer(opt.data_root, opt.h5_file, opt.embed_type, kmer=[1, 2, 3, 4], bases='ACGU', train=True)
train_dataloader_a, train_dataloader_c, train_dataloader_g, train_dataloader_u, train_dataloader_moe, valid_dataloader = \
    generate_dataloaders(*generate_datasets(kmers_datasets))

# Print Number of Data
print('Number of Data:')
print(f'Train for Expert_A: {len(train_dataloader_a.dataset)}')
print(f'Train for Expert_C: {len(train_dataloader_c.dataset)}')
print(f'Train for Expert_G: {len(train_dataloader_g.dataset)}')
print(f'Train for Expert_U: {len(train_dataloader_u.dataset)}')
print(f'Train for MOE: {len(train_dataloader_moe.dataset)}')
print(f'Valid: {len(valid_dataloader.dataset)}')
print('----------------------------------------------------------------------------------------------------')

# Expert Model
expert_a_model = model.Expert(embed_dims=data.calc_embed_dims(opt.embed_type))
expert_c_model = model.Expert(embed_dims=data.calc_embed_dims(opt.embed_type))
expert_g_model = model.Expert(embed_dims=data.calc_embed_dims(opt.embed_type))
expert_u_model = model.Expert(embed_dims=data.calc_embed_dims(opt.embed_type))

# Expert Optimizer
expert_a_optim = Adam(expert_a_model.parameters(), lr=opt.lr)
expert_c_optim = Adam(expert_c_model.parameters(), lr=opt.lr)
expert_g_optim = Adam(expert_g_model.parameters(), lr=opt.lr)
expert_u_optim = Adam(expert_u_model.parameters(), lr=opt.lr)

# Loss
criterion = nn.CrossEntropyLoss()

# Train Expert Model
print('Train Expert A:')
ts.train(expert_a_model, opt.epochs, opt.early_stop, train_dataloader_a, valid_dataloader, criterion, expert_a_optim, opt.save_a, opt.a_logs, opt.device)
print('----------------------------------------------------------------------------------------------------')

print('Train Expert C:')
ts.train(expert_c_model, opt.epochs, opt.early_stop, train_dataloader_c, valid_dataloader, criterion, expert_c_optim, opt.save_c, opt.c_logs, opt.device)
print('----------------------------------------------------------------------------------------------------')

print('Train Expert G:')
ts.train(expert_g_model, opt.epochs, opt.early_stop, train_dataloader_g, valid_dataloader, criterion, expert_g_optim, opt.save_g, opt.g_logs, opt.device)
print('----------------------------------------------------------------------------------------------------')

print('Train Expert U')
ts.train(expert_u_model, opt.epochs, opt.early_stop, train_dataloader_u, valid_dataloader, criterion, expert_u_optim, opt.save_u, opt.u_logs, opt.device)
print('----------------------------------------------------------------------------------------------------')

# Load the parameters of the trained expert models
expert_a_model.load_state_dict(torch.load(opt.save_a, map_location=opt.device))
expert_c_model.load_state_dict(torch.load(opt.save_c, map_location=opt.device))
expert_g_model.load_state_dict(torch.load(opt.save_g, map_location=opt.device))
expert_u_model.load_state_dict(torch.load(opt.save_u, map_location=opt.device))

# MOE Model
moe_model = model.MOE(trained_experts=[expert_a_model, expert_c_model, expert_g_model, expert_u_model],
                      embed_dims=data.calc_embed_dims(opt.embed_type))

# MOE Optimizer
moe_optim = Adam(moe_model.parameters(), lr=opt.lr)

# Train MOE Model
print('Train MOE:')
ts.train(moe_model, opt.epochs, opt.early_stop, train_dataloader_moe, valid_dataloader, criterion, moe_optim, opt.save_moe, opt.moe_logs, opt.device)
print('----------------------------------------------------------------------------------------------------')
