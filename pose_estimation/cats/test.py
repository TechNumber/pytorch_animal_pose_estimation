import math
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_loading.animal_keypoints_dataset import AnimalKeypointsDataset
from utils.transforms import  RandomRotation, RandomFlip, RandomRatioCrop
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from models.lenet_128 import LeNet128
from utils.losses import MSECELoss
from set_random_seed import set_random_seed, SEED
from visualization.keypoints import show_keypoints


def test(model, data_test_loader):

    # set_random_seed(SEED)

    loss = MSECELoss().cuda()
    loss_value_list = []

    for batch, batch_data in enumerate(data_test_loader):
        test_img = batch_data['image'].cuda()
        test_kp = batch_data['keypoints'].cuda()

        pred_kp = model(test_img)
        pred_kp = pred_kp.view(pred_kp.shape[0], 16, 3)
        test_kp = test_kp.view(test_kp.shape[0], 16, 3)

        loss_value = loss(pred_kp, test_kp).item()
        loss_value_list.append(loss_value)

    # t = transforms.ToPILImage()
    # plt.figure()
    # show_keypoints(
    #     t(test_img[-1].cpu()),
    #     pred_kp[-1].view(16, 3).squeeze().detach().cpu(),
    #     show_edges=True
    # )

    # fig = plt.figure(figsize=(5, 5))
    # for i in range(4):
    #     sample = data_test[random.randint(0, len(data_test) - 1)]
    #
    #     test_img = sample['image'].cuda()
    #     test_img = test_img.unsqueeze(0)
    #     pred_kp = model(test_img)
    #     pred_kp = pred_kp.view(pred_kp.shape[0], 16, 3)
    #
    #     ax = plt.subplot(2, 2, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #     show_keypoints(
    #         t(test_img[-1].cpu()),
    #         pred_kp[-1].view(16, 3).squeeze().detach().cpu(),
    #         show_edges=i % 2
    #     )
    # plt.show()

    avg_loss_value = np.sum(loss_value_list) / len(loss_value_list)
    return avg_loss_value
