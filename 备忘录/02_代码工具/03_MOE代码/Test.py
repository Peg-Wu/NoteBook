import os

import torch
from tools import data, model, train_script as ts
from torch.utils.data import DataLoader
import sys
import argparse

# Hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=520)
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--h5_file', type=str, default='./embed.h5')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--embed_type', default=['DNABertEmbedder/3mer'])
"""
--embed_type:
'OneHotEmbedder','DNABert2Embedder','DNABertEmbedder/3mer','DNABertEmbedder/4mer','DNABertEmbedder/5mer',
'DNABertEmbedder/6mer','NucleotideTransformerEmbedder','GENALMEmbedder/bigbird','GENALMEmbedder/bert','GROVEREmbedder'
"""
parser.add_argument('--save_moe', type=str, default='./model_param/moe/db3/moe.pkl')
parser.add_argument('--test_logs', type=str, default='./test_logs/odb3.txt')
opt = parser.parse_args()

# Device
opt.device = torch.device(opt.device)

# Seed
ts.same_seed(opt.seed)

# Dataset & Dataloader (Embedding)
# dataset = data.merge_datasets(data.get_datasets(opt.data_root, opt.h5_file, opt.embed_type, bases='ACGU', train=False))
# dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

# Dataset & Dataloader (Kmer Features)
dataset = data.merge_datasets(data.get_datasets_kmer(opt.data_root, opt.h5_file, opt.embed_type, kmer=[1, 2, 3, 4], bases='ACGU', train=False))
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

# Number of Data
print(f"Data Number: ", len(dataset))
print('--------------------------------------------------------------------------------')

# Expert_Model
expert_a_model = model.Expert(embed_dims=data.calc_embed_dims(opt.embed_type))
expert_c_model = model.Expert(embed_dims=data.calc_embed_dims(opt.embed_type))
expert_g_model = model.Expert(embed_dims=data.calc_embed_dims(opt.embed_type))
expert_u_model = model.Expert(embed_dims=data.calc_embed_dims(opt.embed_type))
expert_models = [expert_a_model, expert_c_model, expert_g_model, expert_u_model]

# MOE_Model
moe_model = model.MOE(trained_experts=expert_models, embed_dims=data.calc_embed_dims(opt.embed_type))
moe_model.load_state_dict(torch.load(opt.save_moe, map_location=opt.device))

# Logs
os.makedirs(os.path.split(opt.test_logs)[0], exist_ok=True)
sys.stdout = ts.Logger(opt.test_logs)
sys.stderr = ts.Logger(opt.test_logs)

# Test Expert A
print("Expert A:")
ts.test(moe_model.experts[0], dataloader, opt.device)

# Test Expert C
print("Expert C:")
ts.test(moe_model.experts[1], dataloader, opt.device)

# Test Expert G
print("Expert G:")
ts.test(moe_model.experts[2], dataloader, opt.device)

# Test Expert U
print("Expert U:")
ts.test(moe_model.experts[3], dataloader, opt.device)

# Test MOE
print("MOE:")
ts.test(moe_model, dataloader, opt.device)
