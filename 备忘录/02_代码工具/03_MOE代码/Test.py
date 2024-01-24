import torch
from utils import data, model
from utils import train_script as ts

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
seed = 520
batch_size = 128
test_data_path = './data/2OM_Test/csv'
Dataset_type = [data.Dataset_onehot]
# Dataset_type = [data.Dataset_word2vec, 64]
in_channels = 4  # 原始序列的特征维度
pin_memory = True

# Seed
ts.same_seed(520)

# Dataset & Dataloader
dataset, dataloader = data.get_test_dataset_dataloader(test_data_path, batch_size, *Dataset_type, pin_memory=pin_memory)

# 数据量
print(f"Data Number: ", len(dataset))

# Expert_Model
Expert_A = model.Expert(in_channels).to(device)
Expert_A.load_state_dict(torch.load(f"./model_param/A.pkl", map_location=device))
Expert_G = model.Expert(in_channels).to(device)
Expert_G.load_state_dict(torch.load(f"./model_param/G.pkl", map_location=device))
Expert_C = model.Expert(in_channels).to(device)
Expert_C.load_state_dict(torch.load(f"./model_param/C.pkl", map_location=device))
Expert_U = model.Expert(in_channels).to(device)
Expert_U.load_state_dict(torch.load(f"./model_param/U.pkl", map_location=device))
expert_models = [Expert_A, Expert_G, Expert_C, Expert_U]

# MOE_Model
MOE = model.MOE(trained_experts=expert_models).to(device)
MOE.load_state_dict(torch.load(f"./model_param/MOE.pkl", map_location=device))

# Test
def test(test_model, test_dataloader):
    accumulator = ts.Accumulator(4)
    test_model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            output = test_model(X)
            accumulator.add(*ts.evaluate(output, y))
    metric = ts.Metrics(*accumulator.data)
    print(f'Accuracy: {metric.accuracy()}')
    print(f'Precision: {metric.precision()}')
    print(f'F1: {metric.f1()}')
    print(f'MCC: {metric.mcc()}')
    print(f'Sn: {metric.sn()}')
    print(f'Sp: {metric.sp()}')
    print('--------------------------------------------------')

# Test_Expert_A
print("Expert_A:")
test(Expert_A, dataloader)
# Test_Expert_G
print("Expert_G:")
test(Expert_G, dataloader)
# Test_Expert_C
print("Expert_C:")
test(Expert_C, dataloader)
# Test_Expert_U
print("Expert_U:")
test(Expert_U, dataloader)
# Test_MOE
print("MOE:")
test(MOE, dataloader)
