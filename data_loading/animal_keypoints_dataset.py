import torch
import json
import os.path
import numpy as np

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
        keypoints = keypoints.astype('float32').reshape(-1, 3)
        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            if 'all' in self.transform.keys() and self.transform['all'] is not None:
                sample = self.transform['all'](sample)
            if 'image' in self.transform.keys() and self.transform['image'] is not None:
                sample['image'] = self.transform['image'](sample['image'])
            if 'keypoints' in self.transform.keys() and self.transform['keypoints'] is not None:
                sample['keypoints'] = self.transform['keypoints'](sample['keypoints'])

        return sample
