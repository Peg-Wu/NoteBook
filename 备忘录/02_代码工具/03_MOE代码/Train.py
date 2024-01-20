from utils import data, model
from torch import nn
import torch
from torch.utils.data import random_split

# Get Dataset & DataLoader || After Onehot-encoding & Positional-encoding
expert_dataset, expert_dataloader, moe_dataset, moe_dataloader = data.main(ratio=0.8)
expert_dataset_A, expert_dataset_G, expert_dataset_C, expert_dataset_U = expert_dataset
expert_dataloader_A, expert_dataloader_G, expert_dataloader_C, expert_dataloader_U = expert_dataloader

# Print Data Number
print(f'A_for_expert:', len(expert_dataset_A))
print(f'G_for_expert:', len(expert_dataset_G))
print(f'C_for_expert:', len(expert_dataset_C))
print(f'U_for_expert:', len(expert_dataset_U))
print(f'all_for_moe:', len(moe_dataset))

# Hyperparameters
epochs = 5
lr = 0.001

# Model
Expert_A = model.Expert().to('cuda')
Expert_G = model.Expert().to('cuda')
Expert_C = model.Expert().to('cuda')
Expert_U = model.Expert().to('cuda')

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
Expert_A_Optim = torch.optim.Adam(Expert_A.parameters(), lr)
Expert_G_Optim = torch.optim.Adam(Expert_G.parameters(), lr)
Expert_C_Optim = torch.optim.Adam(Expert_C.parameters(), lr)
Expert_U_Optim = torch.optim.Adam(Expert_U.parameters(), lr)

# Train Expert_A
for epoch in range(epochs):
    for X, y in expert_dataloader_A:
        X, y = X.to('cuda'), y.to('cuda')
        Expert_A.train()
        Expert_A_Optim.zero_grad()
        outputs = Expert_A(X)
        loss = criterion(outputs, y)
        loss.backward()
        Expert_A_Optim.step()
    print(f'Expert_A is training!, Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

# Train Expert_G
for epoch in range(epochs):
    for X, y in expert_dataloader_G:
        X, y = X.to('cuda'), y.to('cuda')
        Expert_G.train()
        Expert_G_Optim.zero_grad()
        outputs = Expert_G(X)
        loss = criterion(outputs, y)
        loss.backward()
        Expert_G_Optim.step()
    print(f'Expert_G is training!, Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

# Train Expert_C
for epoch in range(epochs):
    for X, y in expert_dataloader_C:
        X, y = X.to('cuda'), y.to('cuda')
        Expert_C.train()
        Expert_C_Optim.zero_grad()
        outputs = Expert_C(X)
        loss = criterion(outputs, y)
        loss.backward()
        Expert_C_Optim.step()
    print(f'Expert_C is training!, Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

# Train Expert_U
for epoch in range(epochs):
    for X, y in expert_dataloader_U:
        X, y = X.to('cuda'), y.to('cuda')
        Expert_U.train()
        Expert_U_Optim.zero_grad()
        outputs = Expert_U(X)
        loss = criterion(outputs, y)
        loss.backward()
        Expert_U_Optim.step()
    print(f'Expert_U is training!, Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

# MOE Model
MOE = model.MOE(trained_experts=[Expert_A, Expert_G, Expert_C, Expert_U]).to('cuda')

# MOE Optimizer
MOE_Optim = torch.optim.Adam(params=MOE.parameters(), lr=lr)

# Train MOE
for epoch in range(epochs):
    for X, y in moe_dataloader:
        X, y = X.to('cuda'), y.to('cuda')
        MOE.train()
        MOE_Optim.zero_grad()
        outputs = MOE(X)
        loss = criterion(outputs, y)
        loss.backward()
        MOE_Optim.step()
    print(f'MOE is training!, Epoch: {epoch + 1}, Loss: {loss.item():.4f}')


# 用测试集测测准确率
# Evaluate all models
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / len(y)
    return accuracy

_, dataloader = data.get_test()
for X, y in dataloader:
    X, y = X.to('cuda'), y.to('cuda')
    acc_A = evaluate(Expert_A, X, y)
    acc_G = evaluate(Expert_G, X, y)
    acc_C = evaluate(Expert_C, X, y)
    acc_U = evaluate(Expert_U, X, y)
    acc_MOE = evaluate(MOE, X, y)

    print("Expert_A Accuracy:", acc_A)
    print("Expert_G Accuracy:", acc_G)
    print("Expert_C Accuracy:", acc_C)
    print("Expert_U Accuracy:", acc_U)
    print("Mixture of Experts Accuracy:", acc_MOE)

