import torch
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter

def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate(output, y):
    """
    该函数用于二分类问题计算TP、TN、FP、FN
    :param output: 预测值
    :param y: 真实值
    :return: 列表 [TP, TN, FP, FN]
    """
    assert type(output) == torch.Tensor and type(y) == torch.Tensor
    with torch.no_grad():
        output = output.detach().argmax(dim=-1).long()
        y = y.detach().long().reshape(output.shape)
        TP = ((output == 1) & (y == 1)).sum().item()
        TN = ((output == 0) & (y == 0)).sum().item()
        FP = ((output == 1) & (y == 0)).sum().item()
        FN = ((output == 0) & (y == 1)).sum().item()
    return [TP, TN, FP, FN]

class Metrics:
    def __init__(self, TP, TN, FP, FN, eps=1e-6):
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN
        self.eps = eps

    def accuracy(self):
        return (self.TP + self.TN) / \
               (self.TP + self.TN + self.FP + self.FN + self.eps)

    def mcc(self):
        return (self.TP * self.TN - self.FP * self.FN) / \
               math.sqrt((self.TP + self.FP + self.eps) * (self.TP + self.FN + self.eps) *
                         (self.TN + self.FP + self.eps) * (self.TN + self.FN + self.eps))

    def sn(self):
        return self.TP / (self.TP + self.FN + self.eps)

    def sp(self):
        return self.TN / (self.TN + self.FP + self.eps)

    def precision(self):
        return self.TP / (self.TP + self.FP + self.eps)

    def f1(self):
        return (2 * self.TP) / (2 * self.TP + self.FP + self.FN + self.eps)

def train_for_experts(model, model_save_path,
                      epochs,
                      dataloader_train,
                      loss_fn, optimizer,
                      device, early_stop=5, dataloader_valid=None):
    """该函数用于训练专家模型"""
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    best_loss, early_stop_count = math.inf, 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X, y in dataloader_train:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(dataloader_train)
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}")

        # Early_Stop
        if epoch_loss <= best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Saving model with loss {best_loss:.4f}')
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop:
            print(f'Model is not improving in {early_stop} epochs, so we halt the training session.')
            break

def train_for_moe(model, model_save_path,
                  epochs,
                  dataloader_train, dataloader_valid,
                  loss_fn, optimizer, device, early_stop=5):
    """该函数用于训练MOE模型"""
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    best_loss, early_stop_count = math.inf, 0
    writer = SummaryWriter(log_dir="./logs")
    accumulator = Accumulator(4)  # TP, TN, FP, FN
    for epoch in range(epochs):
        epoch_loss = 0.0
        accumulator.reset()
        model.train()
        for X, y in dataloader_train:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            accumulator.add(*evaluate(output, y))  # 计算TP、TN、FP、FN并累积
            loss = loss_fn(output, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(dataloader_train)
        writer.add_scalar("MOE_Train_Loss", epoch_loss, epoch + 1)
        # 计算各指标
        calc = Metrics(*accumulator.data)
        epoch_acc = calc.accuracy()
        epoch_mcc = calc.mcc()
        epoch_f1 = calc.f1()
        epoch_sn = calc.sn()
        epoch_sp = calc.sp()
        epoch_pre = calc.precision()
        # 打印损失和准确率 / epoch
        print(f"Train_Process, Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        # 添加到tensorboard
        writer.add_scalar("MOE_Train_ACC", epoch_acc, epoch + 1)
        writer.add_scalar("MOE_Train_MCC", epoch_mcc, epoch + 1)
        writer.add_scalar("MOE_Train_F1", epoch_f1, epoch + 1)
        writer.add_scalar("MOE_Train_SN", epoch_sn, epoch + 1)
        writer.add_scalar("MOE_Train_SP", epoch_sp, epoch + 1)
        writer.add_scalar("MOE_Train_PRE", epoch_pre, epoch + 1)

        # 每训练一个epoch验证一次
        eval_epoch_loss = 0.0
        accumulator.reset()
        model.eval()
        with torch.no_grad():
            for X, y in dataloader_valid:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = loss_fn(output, y)
                eval_epoch_loss += loss.item()
                accumulator.add(*evaluate(output, y))  # 计算TP、TN、FP、FN并累积

            eval_epoch_loss /= len(dataloader_valid)
            writer.add_scalar("MOE_Valid_Loss", eval_epoch_loss, epoch + 1)
            # 计算各指标
            calc = Metrics(*accumulator.data)
            epoch_acc = calc.accuracy()
            epoch_mcc = calc.mcc()
            epoch_f1 = calc.f1()
            epoch_sn = calc.sn()
            epoch_sp = calc.sp()
            epoch_pre = calc.precision()
            # 打印损失和准确率 / epoch
            print(f"Valid_Process, Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            # 添加到tensorboard
            writer.add_scalar("MOE_Valid_ACC", epoch_acc, epoch + 1)
            writer.add_scalar("MOE_Valid_MCC", epoch_mcc, epoch + 1)
            writer.add_scalar("MOE_Valid_F1", epoch_f1, epoch + 1)
            writer.add_scalar("MOE_Valid_SN", epoch_sn, epoch + 1)
            writer.add_scalar("MOE_Valid_SP", epoch_sp, epoch + 1)
            writer.add_scalar("MOE_Valid_PRE", epoch_pre, epoch + 1)

        # Early_Stop (作用在验证集上)
        if eval_epoch_loss <= best_loss:
            best_loss = eval_epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Saving model with loss {best_loss:.4f}')
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop:
            print(f'Model is not improving in {early_stop} epochs, so we halt the training session.')
            break
    writer.close()

if __name__ == '__main__':
    pass