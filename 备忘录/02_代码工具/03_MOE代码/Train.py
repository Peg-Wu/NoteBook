import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
from tools import data, train_script as ts, model
import argparse

# Hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--seed', type=int, default=520)
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--h5_file', type=str, default='./embed.h5')
parser.add_argument('--train_ratio', type=float, default=0.7, help='train_dataset_ratio')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--early_stop', type=int, default=10)
parser.add_argument('--embed_type', default=['OneHotEmbedder'])
"""
--embed_type:
'OneHotEmbedder','DNABert2Embedder','DNABertEmbedder/3mer','DNABertEmbedder/4mer','DNABertEmbedder/5mer',
'DNABertEmbedder/6mer','NucleotideTransformerEmbedder','GENALMEmbedder/bigbird','GENALMEmbedder/bert','GROVEREmbedder'
"""

# TODO: Add Model Parameters
pass

parser.add_argument('--a_logs', type=str, default='./train_logs/a/oh')
parser.add_argument('--c_logs', type=str, default='./train_logs/c/oh')
parser.add_argument('--g_logs', type=str, default='./train_logs/g/oh')
parser.add_argument('--u_logs', type=str, default='./train_logs/u/oh')
parser.add_argument('--moe_logs', type=str, default='./train_logs/moe/oh')

parser.add_argument('--save_a', type=str, default='./model_param/a/oh/a.pkl')
parser.add_argument('--save_c', type=str, default='./model_param/c/oh/c.pkl')
parser.add_argument('--save_g', type=str, default='./model_param/g/oh/g.pkl')
parser.add_argument('--save_u', type=str, default='./model_param/u/oh/u.pkl')
parser.add_argument('--save_moe', type=str, default='./model_param/moe/oh/moe.pkl')
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

# Dataset
datasets = data.get_datasets(opt.data_root, opt.h5_file, opt.embed_type, bases='ACGU', train=True)
datasets_split = []
for dataset in datasets:
    datasets_split.append(data.split_dataset_2parts(dataset, opt.train_ratio))
train_dataset_moe, valid_dataset = \
    data.merge_datasets([each[0] for each in datasets_split]), \
    data.merge_datasets([each[1] for each in datasets_split])
train_dataset_a, train_dataset_c, train_dataset_g, train_dataset_u = \
    [each[0] for each in datasets_split]

# DataLoader
train_dataloader_a = DataLoader(train_dataset_a, batch_size=opt.batch_size, shuffle=True)
train_dataloader_c = DataLoader(train_dataset_c, batch_size=opt.batch_size, shuffle=True)
train_dataloader_g = DataLoader(train_dataset_g, batch_size=opt.batch_size, shuffle=True)
train_dataloader_u = DataLoader(train_dataset_u, batch_size=opt.batch_size, shuffle=True)
train_dataloader_moe = DataLoader(train_dataset_moe, batch_size=opt.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)

# Print Number of Data
print('Number of Data:')
print(f'Train for Expert_A: {len(train_dataset_a)}')
print(f'Train for Expert_C: {len(train_dataset_c)}')
print(f'Train for Expert_G: {len(train_dataset_g)}')
print(f'Train for Expert_U: {len(train_dataset_u)}')
print(f'Train for MOE: {len(train_dataset_moe)}')
print(f'Valid: {len(valid_dataset)}')
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
