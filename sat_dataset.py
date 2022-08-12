import os

import PIL.Image
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import os
import random
from sklearn.preprocessing import MinMaxScaler


class SatDataset(Dataset):
    def __init__(self, is_train=True):
        self.img_dir = "data/train"
        if is_train is False:
            self.img_dir = "data/test"
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop((2304,2304)),
            transforms.Resize(64)
        ])
        ages = os.listdir(self.img_dir)
        self.image_list = []
        self.age_list = []
        i = 0
        for age in ages:
            images = os.listdir(self.img_dir+"/"+age)
            for image in images:
                self.image_list.append(image)
                self.age_list.append(age)
                i = i + 1

        self.__scale__()

    def __scale__(self):
        labels = [[float(i)] for i in self.age_list]
        self.scaler = MinMaxScaler()
        labels = self.scaler.fit_transform(labels)
        labels = torch.tensor(labels, dtype=torch.float32)
        self.age_torch_list = torch.squeeze(labels)

    def unscale(self, values):
        values = [[i] for i in values]
        values = self.scaler.inverse_transform(values)
        values = [i[0] for i in values]
        return values

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        age = self.age_list[idx]
        age_torch = self.age_torch_list[idx]
        img_path = os.path.join(self.img_dir, age, image_name)
        image = PIL.Image.open(img_path)
        image = self.transforms(image)
        return image, age_torch

if __name__ == "__main__":
    cid = SatDataset()
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)

    print(cid.age_list)
    print(cid.age_torch_list)
    print("unscaled")
    print(cid.unscale([100]))
    print(cid.unscale([1,2,100,200]))

    for image, label in dataloader:
        print(image.shape)
        print(label)
        for i in image:
            plt.imshow(i[0].numpy())
            plt.show()
            exit(0)

