from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


# 输入    *PIL         *Image.open()
# 输出    *tensor      *ToTensor()
# 作用    *ndarray     *cv.imread()

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer = SummaryWriter("logs")
writer.add_image("totensor", img_tensor)

# Normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

writer.close()