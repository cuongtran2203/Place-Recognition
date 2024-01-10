import os
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch

class Tokyo247_Dataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [i for i in image_paths if 'forg' not in i]
        self.labels_list = [i.split(os.sep)[-2] for i in self.image_paths]
        self.labels = list(set(self.labels_list))
        print('Number of distince classes:', len(self.labels))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.image_paths[idx]
        img = cv2.imread(path)
        y = self.labels.index(path.split(os.sep)[-2])
        
        if self.transform:
            img = self.transform(image=img)['image']
        return img, y