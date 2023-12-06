from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

img_path = "./dataset/train/ants/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

writer = SummaryWriter("logs")
writer.add_image("test", img_array, 1, dataformats="HWC")
writer.close()