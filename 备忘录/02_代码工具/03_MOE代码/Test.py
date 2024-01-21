import torch
from utils import data, model
from utils import train_script as ts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 因为测试数据的数据量不大，所以dataloader的batch_size设置成了len(dataset)，即一次load出所有的数据
# 如果内存不够，需要修改，请进入./utils/data中修改get_test函数
dataset, dataloader = data.get_test()

# Expert_Model
Expert_A = model.Expert().to(device)
Expert_A.load_state_dict(torch.load(f"./model_param/A.pkl", map_location=device))
Expert_G = model.Expert().to(device)
Expert_G.load_state_dict(torch.load(f"./model_param/G.pkl", map_location=device))
Expert_C = model.Expert().to(device)
Expert_C.load_state_dict(torch.load(f"./model_param/C.pkl", map_location=device))
Expert_U = model.Expert().to(device)
Expert_U.load_state_dict(torch.load(f"./model_param/U.pkl", map_location=device))
expert_models = [Expert_A, Expert_G, Expert_C, Expert_U]

# MOE_Model
MOE = model.MOE(trained_experts=expert_models).to(device)
MOE.load_state_dict(torch.load(f"./model_param/MOE.pkl", map_location=device))

# Test
def test(test_model, test_dataloader):
    accumulator = ts.Accumulator(4)
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
