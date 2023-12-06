from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

trans = transforms.Compose([transforms.Resize((512, 512)),
                            transforms.RandomCrop(128),
                            transforms.ToTensor()])

writer = SummaryWriter("logs")
for i in range(10):
    img_trans = trans(img)
    writer.add_image("new", img_trans, i)

writer.close()