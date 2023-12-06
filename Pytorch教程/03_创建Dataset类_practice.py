from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_path = os.path.join(self.root_dir, self.img_dir)
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        # 因为label和image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_item_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_item_path)

        label_name = self.label_list[idx]
        label_item_path = os.path.join(self.label_path, label_name)
        with open(label_item_path, 'r') as f:
            label = f.readline()
        return img, label

    def __len__(self):
        return len(self.image_list)

root_dir = "./practice/train"
img_ants_dir = "ants_image"
img_bees_dir = "bees_image"
label_ants_dir = "ants_label"
label_bees_dir = "bees_label"

ants_dataset = MyData(root_dir, img_ants_dir, label_ants_dir)
bees_dataset = MyData(root_dir, img_bees_dir, label_bees_dir)
train_dataset = ants_dataset + bees_dataset