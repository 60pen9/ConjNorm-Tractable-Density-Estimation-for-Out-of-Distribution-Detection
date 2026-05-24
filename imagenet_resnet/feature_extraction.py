import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from PIL import Image
from dataloader import CustomDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset()
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=512,
                                          shuffle=False,
                                          num_workers=8)

# load model
# https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
import torch.utils.model_zoo as model_zoo
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

model.eval()
features = list(model.children())[:-1]
modelout = nn.Sequential(*features).to(device)



image_name_list = ()
image_feature_list = None
reset = False
for index, (imgs, labels) in enumerate(tqdm(data_loader)):
    imgs = imgs.to(device)
    out = modelout(imgs)
    out = out.squeeze().cpu().detach().numpy()
    image_name_list = image_name_list + labels
    if index == 0 or reset is True:
        reset = False
        image_feature_list = out
    else:
        image_feature_list = np.concatenate((image_feature_list, out))
    # print((index * 256) / 1281167)

    # save data
    if index != 0 and index % 100 == 0:
        d = {'names': np.asarray(image_name_list), 'features': np.asarray(image_feature_list)}
        np.savez(f"train_features_{str(index).zfill(4)}", **d)
        # np.savez(f"val_features_{str(index).zfill(4)}", **d)
        image_name_list = ()
        image_feature_list = None
        reset = True

d = {'names': np.asarray(image_name_list), 'features': np.asarray(image_feature_list)}
np.savez(f"train_features_{str(index).zfill(4)}", **d)
# np.savez(f"val_features_{str(index).zfill(4)}", **d)

# # read data
# container = np.load(f"features_{index}" + ".npz")
# e = {name: container[name] for name in container}

print("FINISH.")
