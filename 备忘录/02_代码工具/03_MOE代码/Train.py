from utils import data, model
from utils import train_script as ts
from torch import nn
import torch
from torch.utils.data import random_split

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 520
epochs = 1  # 专家模型的epochs，训练集loss在early_stop个epochs之后没有降低，则提前停止训练
early_stop = 5  # 专家模型和MOE模型共用
epochs_moe = 1  # MOE模型的epochs，验证集loss在early_stop个epochs之后没有降低，则提前停止训练
lr = 0.001  # 专家模型的learning_rate
decay = 0.1  # MOE模型的learning_rate是专家模型的decay倍
save_a = "./model_param/A.pkl"  # Expert_A模型保存路径
save_g = "./model_param/G.pkl"  # Expert_G模型保存路径
save_c = "./model_param/C.pkl"  # Expert_C模型保存路径
save_u = "./model_param/U.pkl"  # Expert_U模型保存路径
save_moe = "./model_param/MOE.pkl"  # MOE模型保存路径

# Seed
ts.same_seed(seed)

# Get Dataset & DataLoader || After Onehot-encoding & Positional-encoding
expert_dataset, expert_dataloader, moe_dataset, moe_dataloader = data.main(ratio=0.8, moe_for_train=0.7)
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
Expert_A = model.Expert()
Expert_G = model.Expert()
Expert_C = model.Expert()
Expert_U = model.Expert()
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
ts.train_for_experts(Expert_A, save_a, epochs, expert_dataloader_A, criterion, Expert_A_Optim, device, early_stop)
print('--------------------------------------------------')

# Train Expert_G
print(f"Train Expert_G Model:")
ts.train_for_experts(Expert_G, save_g, epochs, expert_dataloader_G, criterion, Expert_G_Optim, device, early_stop)
print('--------------------------------------------------')

# Train Expert_C
print(f"Train Expert_C Model:")
ts.train_for_experts(Expert_C, save_c, epochs, expert_dataloader_C, criterion, Expert_C_Optim, device, early_stop)
print('--------------------------------------------------')

# Train Expert_U
print(f"Train Expert_U Model:")
ts.train_for_experts(Expert_U, save_u, epochs, expert_dataloader_U, criterion, Expert_U_Optim, device, early_stop)
print('--------------------------------------------------')

# MOE_Model
MOE = model.MOE(trained_experts=expert_models)

# MOE_Optimizer
MOE_Optim = torch.optim.Adam(params=MOE.parameters(), lr=lr * decay)

# Train MOE
print(f"Train MOE Model:")
ts.train_for_moe(MOE, save_moe, epochs_moe,
                 moe_dataloader_train, moe_dataloader_valid, criterion, MOE_Optim, device, early_stop)
print('--------------------------------------------------')
