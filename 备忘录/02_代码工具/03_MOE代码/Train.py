from utils import data, model
from utils import train_script as ts
from torch import nn
import torch
from torch.utils.data import random_split

# Hyperparameters
seed = 520
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
Dataset_type = [data.Dataset_onehot]  # 原始序列通过onehot编码
# Dataset_type = [data.Dataset_word2vec, 64]  # 原始序列经过词嵌入编码, 后面的数字是embed_dim
in_channels = 4  # 原始序列的特征维度
batch_size = 64
epochs = 10  # 专家模型的epochs，训练集loss在early_stop个epochs之后没有降低，则提前停止训练
early_stop = 3  # 专家模型early_stop
epochs_moe = 30  # MOE模型的epochs，验证集accuracy在early_stop个epochs之后没有升高，则提前停止训练
early_stop_moe = 5  # MOE模型early_stop
lr = 0.001  # 专家模型的learning_rate
decay = 0.1  # MOE模型的learning_rate是专家模型的decay倍
ratio = 0.7  # 训练集比例
pin_memory = True
save_a = "./model_param/A.pkl"  # Expert_A模型保存路径
save_g = "./model_param/G.pkl"  # Expert_G模型保存路径
save_c = "./model_param/C.pkl"  # Expert_C模型保存路径
save_u = "./model_param/U.pkl"  # Expert_U模型保存路径
save_moe = "./model_param/MOE.pkl"  # MOE模型保存路径

# Seed
ts.same_seed(seed)

# Get Dataset & DataLoader
A_path = './data/2OM_Train/csv/A2OM_train.csv'
G_path = './data/2OM_Train/csv/G2OM_train.csv'
C_path = './data/2OM_Train/csv/C2OM_train.csv'
U_path = './data/2OM_Train/csv/U2OM_train.csv'
expert_dataset, expert_dataloader, moe_dataset, moe_dataloader = \
    data.get_train_dataset_dataloader(A_path, G_path, C_path, U_path, ratio, batch_size,
                                      *Dataset_type, pin_memory=pin_memory)
expert_dataset_A, expert_dataset_G, expert_dataset_C, expert_dataset_U = expert_dataset
expert_dataloader_A, expert_dataloader_G, expert_dataloader_C, expert_dataloader_U = expert_dataloader
moe_dataset_train, moe_dataset_valid = moe_dataset
moe_dataloader_train, moe_dataloader_valid = moe_dataloader

# Number of Data
print(f'Number of Data:')
print(f'A_for_expert:', len(expert_dataset_A))
print(f'G_for_expert:', len(expert_dataset_G))
print(f'C_for_expert:', len(expert_dataset_C))
print(f'U_for_expert:', len(expert_dataset_U))
print(f'MoE_train:', len(moe_dataset_train))
print(f'MoE_valid:', len(moe_dataset_valid))
print('--------------------------------------------------')

# Expert_Model
Expert_A = model.Expert(in_channels)
Expert_G = model.Expert(in_channels)
Expert_C = model.Expert(in_channels)
Expert_U = model.Expert(in_channels)
expert_models = [Expert_A, Expert_G, Expert_C, Expert_U]

# Loss
criterion = nn.CrossEntropyLoss()

# Expert_Optimizer
Expert_A_Optim = torch.optim.Adam(Expert_A.parameters(), lr)
Expert_G_Optim = torch.optim.Adam(Expert_G.parameters(), lr)
Expert_C_Optim = torch.optim.Adam(Expert_C.parameters(), lr)
Expert_U_Optim = torch.optim.Adam(Expert_U.parameters(), lr)
expert_optimizers = [Expert_A_Optim, Expert_G_Optim, Expert_C_Optim, Expert_U_Optim]

# Train Expert_A
print(f"Train Expert_A Model:")
ts.train_for_experts(Expert_A, epochs, expert_dataloader_A, criterion, Expert_A_Optim, device, early_stop)
print('--------------------------------------------------')

# Train Expert_G
print(f"Train Expert_G Model:")
ts.train_for_experts(Expert_G, epochs, expert_dataloader_G, criterion, Expert_G_Optim, device, early_stop)
print('--------------------------------------------------')

# Train Expert_C
print(f"Train Expert_C Model:")
ts.train_for_experts(Expert_C, epochs, expert_dataloader_C, criterion, Expert_C_Optim, device, early_stop)
print('--------------------------------------------------')

# Train Expert_U
print(f"Train Expert_U Model:")
ts.train_for_experts(Expert_U, epochs, expert_dataloader_U, criterion, Expert_U_Optim, device, early_stop)
print('--------------------------------------------------')

# MOE_Model
MOE = model.MOE(trained_experts=expert_models)

# MOE_Optimizer
MOE_Optim = torch.optim.Adam(params=MOE.parameters(), lr=lr * decay)

# Train MOE
print(f"Train MOE Model:")
ts.train_for_moe(MOE, save_moe, save_a, save_g, save_c, save_u, epochs_moe,
                 moe_dataloader_train, moe_dataloader_valid, criterion, MOE_Optim, device, early_stop_moe)
print('--------------------------------------------------')
