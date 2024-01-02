import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import os

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
random_size = 64
img_size = 1 * 28 * 28
hidden_size = 256
epochs = 200
lr = 0.0001
batch_size = 128
img_path = './generated_images'

# Data
img_process = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = datasets.MNIST(root='./data/MNIST', train=True, transform=img_process, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Model
G = nn.Sequential(nn.Linear(random_size, hidden_size),
                  nn.ReLU(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.ReLU(),
                  nn.Linear(hidden_size, img_size),
                  nn.Tanh()).to(device)

D = nn.Sequential(nn.Linear(img_size, hidden_size),
                  nn.LeakyReLU(0.2),
                  nn.Linear(hidden_size, hidden_size),
                  nn.LeakyReLU(0.2),
                  nn.Linear(hidden_size, 1),
                  nn.Sigmoid()).to(device)

# Loss Function
loss_fn = nn.BCELoss(reduction='sum').to(device)

# Optimizer
optim_G = torch.optim.Adam(G.parameters(), lr=lr)
optim_D = torch.optim.Adam(D.parameters(), lr=lr)

# Path for saving generated images
os.makedirs(img_path, exist_ok=True)

# Train Process
writer = SummaryWriter('./logs')
true_label = torch.ones(batch_size, 1, requires_grad=False).to(device)
fake_label = torch.zeros(batch_size, 1, requires_grad=False).to(device)
for epoch in range(epochs):
    G_loss = 0.0
    D_loss = 0.0
    for imgs, _ in dataloader:
        imgs = imgs.reshape(batch_size, -1).to(device)
        """Train Generator"""
        optim_G.zero_grad()
        random_vector = torch.randn(size=(batch_size, random_size)).to(device)
        fake_imgs = G(random_vector)
        g_loss = loss_fn(D(fake_imgs), true_label)
        g_loss.backward()
        optim_G.step()
        G_loss += g_loss.item()

        """Train Discriminator"""
        optim_D.zero_grad()
        d_loss_true = loss_fn(D(imgs), true_label)
        d_loss_fake = loss_fn(D(fake_imgs.detach()), fake_label)
        d_loss = (d_loss_fake + d_loss_true) / 2
        d_loss.backward()
        optim_D.step()
        D_loss += d_loss.item()

    G_loss /= (len(dataloader) * batch_size)
    D_loss /= (len(dataloader) * batch_size)
    print(f'Epoch:{epoch + 1}, G_Loss:{G_loss:.6f}, D_Loss:{D_loss:.6f}')
    writer.add_scalar('G_Loss / Epoch', G_loss, epoch + 1)
    writer.add_scalar('D_Loss / Epoch', D_loss, epoch + 1)
    writer.add_images('Generated Images / Epoch', fake_imgs.detach().reshape(batch_size, 1, 28, 28), epoch + 1)

    if (epoch + 1) % 5 == 0:
        save_image(fake_imgs.detach().reshape(batch_size, 1, 28, 28)[:25],
                   os.path.join(img_path, f'{epoch + 1}.png'),
                   nrow=5, ncol=5)
writer.close()
