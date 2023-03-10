import json
import os.path
import pathlib
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive

from utils.transforms import RandomRotation, RandomFlip, RandomRatioCrop


class AKD(Dataset):
    _URL = 'https://drive.google.com/file/d/1_pnZohJ4A65uWX7UBTWMTct1lPOLPLl8/view?usp=sharing'
    _MD5 = '1720a863fa30aa6d3321b6d139b42703'

    def __init__(
            self,
            root,
            split='train',
            all_transform=None,
            image_transform=None,
            target_transform=None,
            heatmap_size=None,
            produce_visibility=False,
            download=False,
    ):
        super().__init__()
        self.root = root
        self.split = verify_str_arg(split, 'split', ('train', 'test'))
        self._all_transform = all_transform
        self._target_transform = target_transform
        self._image_transform = image_transform
        if heatmap_size:
            if not isinstance(heatmap_size, (int, Sequence)):
                raise TypeError(f"Size should be int or sequence. Got {type(heatmap_size)}")
            if isinstance(heatmap_size, Sequence) and not len(heatmap_size) == 2:
                raise ValueError("If size is a sequence, it should have 2 values")
            if isinstance(heatmap_size, Sequence):
                self._heatmap_size = heatmap_size
            else:
                self._heatmap_size = (heatmap_size, heatmap_size)
        else:
            self._heatmap_size = None
        self._produce_visibility = produce_visibility

        self._base_folder = pathlib.Path(self.root) / type(self).__name__.lower()
        self._data_folder = self._base_folder / 'cats'
        self._meta_folder = self._data_folder / split / f'{split}_keypoints_annotations.json'
        self._images_folder = self._data_folder / split / 'images'

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._meta_folder, 'r') as f:
            self._json_data = json.load(f)
        self.keypoint_names = list(self._json_data['keypoints']['0'].keys())

    def __len__(self):
        return len(self._json_data['keypoints'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # TODO: replace python iterables with numpy iterables

        idx = str(idx)
        image_name = self._json_data['images'][idx]['file_name']
        image_path = os.path.join(self._images_folder, image_name)
        image = Image.open(fp=image_path).convert('RGB')
        keypoints = np.array(list(self._json_data['keypoints'][idx].values()))
        keypoints = keypoints.astype('float32').reshape(-1, 3)
        sample = {'image': image, 'keypoints': keypoints}

        if self._all_transform:
            sample = self._all_transform(sample)
        if self._image_transform:
            sample['image'] = self._image_transform(sample['image'])
        if self._target_transform:
            sample['keypoints'] = self._target_transform(sample['keypoints'])

        if self._heatmap_size:
            sample['heatmap'] = self._create_heatmap(sample)

        if not self._produce_visibility:
            sample['keypoints'] = sample['keypoints'][:, :2]

        return sample

    def _check_exists(self):
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)

    def _download(self):
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=str(self._base_folder), filename='cats.tar.gz',
                                     md5=self._MD5)

    def _gaussian_peak(self, width, height, mean_x, mean_y, std_x, std_y):
        x = torch.arange(start=0, end=width, dtype=torch.float32)
        y = torch.arange(start=0, end=height, dtype=torch.float32)
        y = y.view(height, -1)
        #  / (2 * np.pi * std_x * std_y)
        return torch.exp(
            -((x - mean_x) ** 2 / std_x ** 2 + (y - mean_y) ** 2 / std_y ** 2) / 2
        )

    def _create_heatmap(self, sample):
        STD = 1.5
        image, keypoints = sample['image'], sample['keypoints']
        nkp = len(keypoints)
        height, width = self._heatmap_size[0], self._heatmap_size[1]
        heatmap = torch.zeros(size=(nkp, height, width), dtype=torch.float32)

        for i in range(nkp):
            mean_x = keypoints[i][0] * width
            mean_y = keypoints[i][1] * height
            gaussian = self._gaussian_peak(
                width, height, mean_x, mean_y, STD, STD
            ) * (-1 if self._produce_visibility and keypoints[i][2] == 0 else 1)
            heatmap[i, :, :] = gaussian

        return heatmap


if __name__ == '__main__':
    all_tform = transforms.Compose([
        RandomFlip(0.5, 0.5),
        RandomRatioCrop(0.1, 0.1, 0.9, 0.9),
        RandomRotation((-30, 30)),
    ])

    img_tform = transforms.Compose([
        transforms.Resize((368, 368)),
        transforms.ToTensor(),
    ])

    t = transforms.ToTensor()

    sample = AKD(
        root='./data',
        split='train',
        download=True,
        all_transform=all_tform,
        image_transform=img_tform,
        # target_transform=t,
        produce_visibility=False,
    ).__getitem__(2)
