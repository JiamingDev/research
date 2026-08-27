import torch
from PIL import Image
from torch.utils.data import Dataset
import os


class mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_list = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)  # 返回的是PIL Image
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.image_list)


root_dir = "train"
ant_label_dir = "ant"
ant_dataset = mydata(root_dir, ant_label_dir)
bee_dataset = mydata(root_dir, "bee")
img, label = ant_dataset[0]
img2, label2 = bee_dataset[0]
# img.show()
# img2.show()
train_dataset = ant_dataset + bee_dataset  # 将两数据集拼接，ant先放，bee后放
print(len(train_dataset))
