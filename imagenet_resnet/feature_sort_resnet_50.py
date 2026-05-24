import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from PIL import Image
from dataloader import CustomDataset

# read data
indexes = [str((i+1)*100).zfill(4) for i in range(25)] + [str(2502)]
total_features = np.zeros((1, 2048))
total_labels = np.zeros((1,))
for index in indexes:
    print(index)
    file = np.load("train_features_{index}.npz".format(index=index))
    # features = file["features"]
    # print(features.shape)
    # total_features = np.concatenate([total_features, features], axis=0)
    labels = file["names"]
    total_labels = np.concatenate([total_labels, labels], axis=0)
    print(labels.shape)

total_labels = total_labels[1:]
print(total_features.shape)
print(total_labels.shape)

my_list = list(total_labels)
from collections import Counter
my_counter = Counter(my_list)
labelmapping = {}
i = 0
for key, value in my_counter.items():
    print("{}: {}".format(key, value))
    labelmapping[i] = key
    i = i + 1
# print(i)
# print(labelmapping)

np.save("./imagenet_feature_resnet_50/imagnet_labels.npy", total_labels)

for j in range(1000):
    total_features_j = np.zeros((1, 2048))
    for index in indexes:
        # print(index)
        file = np.load(f"train_features_{index}.npz".format(index=index))
        features = file["features"]
        labels = file["names"]
        features_j = features[np.where(labels == labelmapping[j])]
        # print(features_j.shape)
        total_features_j = np.concatenate([total_features_j, features_j], axis=0)
    print(j, total_features_j.shape)
    np.save("./imagenet_feature_resnet_50/class_{j}.npy".format(j=j), total_features_j[1:])

val1 = np.load("val_features_0097.npz")
features1 = val1["features"]
np.save(f"./imagenet_feature_resnet_50/val.npy", features1)
print(features1.shape)




