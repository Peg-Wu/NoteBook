import torch
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
import sys

def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

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

def train(model,
          epochs,
          early_stop,
          dataloader_train,
          dataloader_valid,
          loss_fn,
          optimizer,
          model_save_path,
          log_dir,
          device,
          **kwargs):
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    best_acc, early_stop_count = 0.0, 0
    writer = SummaryWriter(log_dir=log_dir)
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
        writer.add_scalar("Train_Loss", epoch_loss, epoch + 1)
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
        writer.add_scalar("Train_ACC", epoch_acc, epoch + 1)
        writer.add_scalar("Train_MCC", epoch_mcc, epoch + 1)
        writer.add_scalar("Train_F1", epoch_f1, epoch + 1)
        writer.add_scalar("Train_SN", epoch_sn, epoch + 1)
        writer.add_scalar("Train_SP", epoch_sp, epoch + 1)
        writer.add_scalar("Train_PRE", epoch_pre, epoch + 1)

        # 每训练一个epoch验证一次
        epoch_loss = 0.0
        accumulator.reset()
        model.eval()
        with torch.no_grad():
            for X, y in dataloader_valid:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = loss_fn(output, y)
                epoch_loss += loss.item()
                accumulator.add(*evaluate(output, y))  # 计算TP、TN、FP、FN并累积

            epoch_loss /= len(dataloader_valid)
            writer.add_scalar("Valid_Loss", epoch_loss, epoch + 1)
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
            writer.add_scalar("Valid_ACC", epoch_acc, epoch + 1)
            writer.add_scalar("Valid_MCC", epoch_mcc, epoch + 1)
            writer.add_scalar("Valid_F1", epoch_f1, epoch + 1)
            writer.add_scalar("Valid_SN", epoch_sn, epoch + 1)
            writer.add_scalar("Valid_SP", epoch_sp, epoch + 1)
            writer.add_scalar("Valid_PRE", epoch_pre, epoch + 1)

        # Early_Stop (作用在验证集上)
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'Saving model with valid accuracy {best_acc:.4f}')
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop:
            print(f'Model is not improving in {early_stop} epochs, so we halt the training session.')
            break
    writer.close()

def test(model, dataloader, device):
    model = model.to(device)
    accumulator = Accumulator(4)
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            accumulator.add(*evaluate(output, y))
    metric = Metrics(*accumulator.data)
    print(f'Accuracy: {metric.accuracy()}')
    print(f'Precision: {metric.precision()}')
    print(f'F1: {metric.f1()}')
    print(f'MCC: {metric.mcc()}')
    print(f'Sn: {metric.sn()}')
    print(f'Sp: {metric.sp()}')
    print('--------------------------------------------------------------------------------')


if __name__ == '__main__':
    pass
