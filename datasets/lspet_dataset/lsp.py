from scipy.io import loadmat
from os import listdir
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset


class LSP(Dataset):

    def __init__(self, path, transforms, image_size, hmap_size=0):
        self.image_size = image_size
        self.hmap_size = hmap_size
        self.path = path
        self.joints = loadmat(self.path + 'joints.mat')['joints'].T
        self.transforms = transforms
        self.get_paths()

    def get_paths(self):
        self.paths = listdir(self.path + 'images/')
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        path = self.path + 'images/' + self.paths[index]
        image, x_scale, y_scale = load_image(path, self.transforms, self.image_size)

        joints, center = get_joints(self.joints[index].T, (x_scale, y_scale), self.image_size)

        mask = torch.tensor([m == 0 for m in self.joints[index].T[:, 2:].reshape(-1)])

        if self.hmap_size == 0:
            return image, joints, center, mask

        if self.hmap_size > 0:
            points = joints.reshape(14, 2) * self.hmap_size
            hmap = gen_hmap(self.hmap_size, self.hmap_size, points.numpy(), s=1.5)
            background = torch.sum(hmap, axis=0)
            hmap = torch.cat([background.unsqueeze(0), hmap], axis=0)
            center = gen_hmap(self.hmap_size, self.hmap_size,
                              [[center[0] * self.hmap_size, center[1] * self.hmap_size]], s=3)
            return image, hmap, center, mask


class LSPet(Dataset):

    def __init__(self, path, transforms, image_size, hmap_size=0):
        self.image_size = image_size
        self.hmap_size = hmap_size
        self.path = path
        self.joints = loadmat(self.path + 'joints.mat')['joints'].T
        self.transforms = transforms
        self.get_paths()

    def get_paths(self):
        self.paths = listdir(self.path + 'images/')
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        path = self.path + 'images/' + self.paths[index]
        image, x_scale, y_scale = load_image(path, self.transforms, self.image_size)

        joints, center = get_joints(self.joints[index], (x_scale, y_scale), self.image_size)

        mask = torch.from_numpy(self.joints[index].T[:, 2:]).reshape(-1).float()

        if self.hmap_size == 0:
            return image, joints, center, mask

        if self.hmap_size > 0:
            points = joints.reshape(14, 2) * self.hmap_size
            hmap = gen_hmap(self.hmap_size, self.hmap_size, points.numpy(), s=1.5)
            background = torch.sum(hmap, axis=0)
            hmap = torch.cat([hmap, background.unsqueeze(0)], axis=0)
            center = gen_hmap(self.hmap_size, self.hmap_size,
                              [[center[0] * self.hmap_size, center[1] * self.hmap_size]], s=3)
            return image, hmap, center, mask


# load and transform images and return the image and the x, y scaling
def load_image(path, transforms, size):
    image = Image.open(path)
    image_size = image.size
    image = transforms(image)
    x_scale = size[0] / image_size[0]
    y_scale = size[1] / image_size[1]
    return image, x_scale, y_scale


# get joints and center points
def get_joints(joints, scales, size):
    x_joints = (joints.T[:, :1] * scales[0]) / size[0]
    y_joints = (joints.T[:, 1:2] * scales[1]) / size[1]
    joints = np.concatenate((x_joints, y_joints), axis=1).reshape(-1)
    joints = torch.from_numpy(joints).float()
    center = [sum(x_joints[x_joints > 0]) / len(x_joints[x_joints > 0]),
              sum(y_joints[y_joints > 0]) / len(y_joints[y_joints > 0])]
    return joints, center


# generate gaussian distribution heatmap
def gaussian(x0, y0, sigma, width, height):
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    return np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)))


# Generate heatmaps for n points
def gen_hmap(height, width, points, s=3):
    npoints = len(points)
    hmap = np.zeros((height, width, npoints), dtype=np.float)
    for i in range(npoints):
        if points[i][1] >= 0 and points[i][0] >= 0:
            if points[i][0] >= height: points[i][0] = height - 1
            if points[i][1] >= width: points[i][1] = width - 1
            hmap[:, :, i] = gaussian(points[i][1], points[i][0], s, height, width)
        else:
            hmap[:, :, i] = np.zeros((height, width))
    return torch.from_numpy(hmap.T).float()
