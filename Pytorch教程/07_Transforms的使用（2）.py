import cv2  # pip install opencv-python
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "./dataset/train/ants/0013035.jpg"
cv_img = cv2.imread(img_path)  # cv_img的类型：ndarray

tensor_trans = transforms.ToTensor()
cv_img_tensor = tensor_trans(cv_img)  # ToTensor: PIL Image or numpy.ndarray

writer = SummaryWriter("logs")
writer.add_image("test", cv_img_tensor, 1)
writer.close()



