import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from PIL import Image
from dataloader import CustomDataset

indexes = [str((i+1)*100) for i in range(100)] + [str(10009)]
total_features = np.zeros((1, 2048))
total_labels = np.zeros((1,))
for index in indexes:
    print(index)
    file = np.load(f"imagenet_feature_mobilenet_v2\\train_features_{index}.npz".format(index=index))
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
print(i)
print(labelmapping)

np.save(f"imagenet_feature_mobilenet_v2\\imagnet_labels.npy", total_labels)

for j in range(1000):
    total_features_j = np.zeros((1, 1280))
    for index in indexes:
        print(index)
        file = np.load(f"imagenet_feature_mobilenet_v2\\train_features_{index}.npz".format(index=index))
        features = file["features"]
        labels = file["names"]
        features_j = features[np.where(labels == labelmapping[j])]
        print(features_j.shape)
        total_features_j = np.concatenate([total_features_j, features_j], axis=0)
        print(total_features_j.shape)
        np.save(f"imagenet_feature_mobilenet_v2\\class_{j}.npy".format(j=j), total_features_j[1:])

val1 = np.load(f"imagenet_feature_mobilenet_v2\\val_features_100.npz")
val2 = np.load(f"imagenet_feature_mobilenet_v2\\val_features_200.npz")
val3 = np.load(f"imagenet_feature_mobilenet_v2\\val_features_300.npz")
val4 = np.load(f"imagenet_feature_mobilenet_v2\\val_features_390.npz")
features1 = val1["features"]
features2 = val2["features"]
features3 = val3["features"]
features4 = val4["features"]
features = np.concatenate([features1, features2, features3, features4], axis=0)
np.save(f"imagenet_feature_mobilenet_v2\\val.npy", features)
print(features.shape)




