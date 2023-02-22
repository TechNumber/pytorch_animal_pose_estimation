import math

import torch
import json
import os.path
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


def gaussian_peak(width, height, mean_x, mean_y, std_x, std_y):
    x = torch.arange(start=0, end=width, dtype=torch.float32)
    y = torch.arange(start=0, end=height, dtype=torch.float32)
    y = y.view(height, -1)
    return torch.exp(
        -((x - mean_x) ** 2 / std_x ** 2 + (y - mean_y) ** 2 / std_y ** 2) / 2
    ) / (2 * np.pi * std_x * std_y)


def create_heatmap(sample):
    STD = 1.5
    image, keypoints = sample['image'], sample['keypoints'][0]
    nkp = len(keypoints)
    if isinstance(image, torch.Tensor):
        height, width = 45, 45
    else:
        height, width = 45, 45
    heatmap = torch.zeros(size=(nkp, height, width), dtype=torch.float32)

    for i in range(nkp):
        mean_x = keypoints[i][0] * width
        mean_y = keypoints[i][1] * height
        heatmap[i, :, :] = gaussian_peak(width, height, mean_x, mean_y, STD, STD)

    return heatmap


class AnimalKeypointsDataset(Dataset):

    def __init__(self, json_file_path, image_dir, transform=None, heatmap=False):
        with open(json_file_path, 'r') as f:
            self.json_data = json.load(f)
        self.keypoint_names = list(self.json_data['keypoints']['0'].keys())
        self.image_dir = image_dir
        self.transform = transform
        self.heatmap = heatmap

    def __len__(self):
        return len(self.json_data['keypoints'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # TODO: replace python iterables with numpy iterables

        idx = str(idx)
        image_name = self.json_data['images'][idx]['file_name']
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(fp=image_path)
        keypoints = list(self.json_data['keypoints'][idx].values())
        keypoints = np.array(keypoints)
        keypoints = keypoints.astype('float32').reshape(-1, 3)
        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            if 'all' in self.transform.keys() and self.transform['all'] is not None:
                sample = self.transform['all'](sample)
            if 'image' in self.transform.keys() and self.transform['image'] is not None:
                sample['image'] = self.transform['image'](sample['image'])
            if 'keypoints' in self.transform.keys() and self.transform['keypoints'] is not None:
                sample['keypoints'] = self.transform['keypoints'](sample['keypoints'])

        if self.heatmap:
            sample['heatmap'] = create_heatmap(sample)

        return sample
