import math
import os
import random
import sys
import traceback
import json
import pdb
import torch
import cv2
import numpy as np
from albumentations.augmentations.functional import rot90
import torchvision
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)



class LandmarkDataset(Dataset):

    def __init__(self,
                 annotations,
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 mode="train",
                 transforms=None
                 ):
        super().__init__()
        self.mode = mode
        self.normalize = normalize
        self.transforms = transforms
        self.data = annotations
        self.lost = []

    def load_sample(self,img_path):
        try:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transforms:
                data = self.transforms(image=image)
                image = data["image"]

            image = img_to_tensor(image, self.normalize)
            return image
        except:
            self.lost.append(img_path)
            #pdb.set_trace()
            return torch.randn((3,380,380))
            #pdb.set_trace()

    def __getitem__(self, index: int):
        landmarks = self.data[index][1]
        landmarks = torch.tensor(landmarks,dtype=torch.float)
        img_path = self.data[index][0]
        img = self.load_sample(img_path)
        return landmarks, img, img_path

    def __len__(self) -> int:
        return len(self.data)

    def reset_seed(self,epoch,seed):
        seed = (epoch + 1) * seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # gpu
        torch.backends.cudnn.deterministic = True


def collate_function(data):
    transposed_data = list(zip(*data))
    lab, img, img_path = transposed_data[0], transposed_data[1], transposed_data[2]
    img = torch.stack(img, 0)
    lab = torch.stack(lab, 0)
    return lab, img, img_path
