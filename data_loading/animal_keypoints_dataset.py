import torch
import json
import os.path
import numpy as np
import random
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset
from math import sqrt, cos, sin, asin, pi


class RandomRotation(object):

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints']

        # TODO: add random center
        angle = random.uniform(self.degrees[0], self.degrees[1])
        rad = angle * pi / 180
        w, h = image.width, image.height
        x_c, y_c = w // 2, h // 2

        for kp in keypoints:
            x, y = kp[0] * w, kp[1] * h
            R = sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
            if (x - x_c) / R > 0:
                kp[0] = (x_c + R * cos(asin((y - y_c) / R) - rad)) / w
                kp[1] = (y_c + R * sin(asin((y - y_c) / R) - rad)) / h
            else:
                kp[0] = (x_c + R * cos(pi - asin((y - y_c) / R) - rad)) / w
                kp[1] = (y_c + R * sin(pi - asin((y - y_c) / R) - rad)) / h

        image = TF.rotate(image, angle, expand=False, center=None)

        return {'image': image, 'keypoints': keypoints}


class RandomFlip(object):

    def __init__(self, ver_cv, hor_cv):
        self.ver_cv = ver_cv
        self.hor_cv = hor_cv

    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints']

        ver_p = random.random()
        hor_p = random.random()

        if ver_p < self.ver_cv:
            keypoints[:, 1] = 1 - keypoints[:, 1]
            image = TF.vflip(image)
        if hor_p < self.hor_cv:
            keypoints[:, 0] = 1 - keypoints[:, 0]
            image = TF.hflip(image)

        return {'image': image, 'keypoints': keypoints}


class RandomRatioCrop(object):

    def __init__(self, max_top_offset, max_left_offset, min_crop_height, min_crop_width):
        self.max_top_offset = max_top_offset
        self.max_left_offset = max_left_offset
        self.min_crop_height = min_crop_height
        self.min_crop_width = min_crop_width

    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints']

        top_offset = random.uniform(0, self.max_top_offset)
        left_offset = random.uniform(0, self.max_left_offset)
        crop_height = random.uniform(self.min_crop_height, 1)
        crop_width = random.uniform(self.min_crop_width, 1)

        keypoints[:, 1] = (keypoints[:, 1] - top_offset) / (crop_height * (1 - top_offset))
        keypoints[:, 0] = (keypoints[:, 0] - left_offset) / (crop_width * (1 - left_offset))

        image = TF.crop(
            image,
            top_offset * image.height,
            left_offset * image.width,
            (image.height * (1 - top_offset)) * crop_height,
            (image.width * (1 - left_offset)) * crop_width
        )

        return {'image': image, 'keypoints': keypoints}


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
