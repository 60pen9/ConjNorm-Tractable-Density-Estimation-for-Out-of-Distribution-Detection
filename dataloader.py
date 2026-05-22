import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from PIL import Image
from tqdm import tqdm
import pickle


class CustomDataset(Dataset):
    def __init__(self):
        self.imgs_path = your own data address
        classes = os.listdir(self.imgs_path)
        classes_list = [os.path.join(self.imgs_path, i) for i in classes]
        # print(file_list)

        if os.path.exists("./data"):
            with open("data", "rb") as fp:  # Unpickling
                self.data = pickle.load(fp)
        else:
            self.data = []
            for class_path in tqdm(classes_list):
                class_name = class_path.split("/")[-1]
                images = os.listdir(class_path)
                for img_path in images:
                    self.data.append([os.path.join(class_path, img_path), class_name])
            with open("data", "wb") as fp:  # Pickling
                pickle.dump(self.data, fp)

        # print(self.data)
        self.class_map = {}
        self.to_tensor = transforms.Compose([
            transforms.Resize((256, 256), 2),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.to_tensor(img)

        return img_tensor, class_name


if __name__ == "__main__":
    dataset = CustomDataset()
