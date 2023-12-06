from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()

# 用命令行启动，设置端口，避免和别人冲突
# tensorboard --logdir=logs --port=6007
