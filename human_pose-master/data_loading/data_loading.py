import json
import os.path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AnimalKeypointsDataset(Dataset):

    def __init__(self, json_file_path, image_dir, transform=None):
        with open(json_file_path, 'r') as f:
            self.json_data = json.load(f)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.json_data['keypoints'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = str(idx)
        image_name = self.json_data['images'][idx]['file_name']
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(fp=image_path)
        keypoints = list(self.json_data['keypoints'][idx].values())
        keypoints = np.array(keypoints)
        keypoints = keypoints.astype('float').reshape(-1, 3)
        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample
