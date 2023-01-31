import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_loading.animal_keypoints_dataset import AnimalKeypointsDataset
from utils.transforms import RandomRotation, RandomFlip, RandomRatioCrop
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from models.lenet_128 import LeNet128
from models.conv_pose_machines import ConvolutionalPoseMachines
from utils.losses import MSECELoss
from set_random_seed import set_random_seed, SEED
from visualization.keypoints import show_keypoints
from pose_estimation.cats.test import test_lenet, test_cpm


def train_lenet(model,
                alpha,
                image_size,
                train_batch_size,
                epochs,
                test_batch_size,
                logging_step=50):
    set_random_seed(SEED)

    t = transforms.ToPILImage()

    all_tform = transforms.Compose([
        RandomFlip(0.5, 0.5),
        RandomRatioCrop(0.15, 0.15, 0.85, 0.85),
        RandomRotation((-40, 40)),
    ])

    img_tform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    data_train = AnimalKeypointsDataset(
        json_file_path='../../dataset/cats/train/keypoints_annotations.json',
        image_dir='../../dataset/cats/train/labeled/',
        transform={'all': all_tform,
                   'image': img_tform,
                   'keypoints': transforms.ToTensor()})
    data_train_loader = DataLoader(data_train, batch_size=train_batch_size, shuffle=True, num_workers=0)

    data_test = AnimalKeypointsDataset(
        json_file_path='../../dataset/cats/test/keypoints_annotations.json',
        image_dir='../../dataset/cats/test/labeled/',
        transform={'all': all_tform,
                   'image': img_tform,
                   'keypoints': transforms.ToTensor()})
    data_test_loader = DataLoader(data_test, batch_size=test_batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    loss = MSECELoss().cuda()
    epoch_train_loss_list = []
    epoch_test_loss_list = []
    min_epoch_test_loss_list = []

    for epoch in range(epochs):
        batch_train_loss_list = []
        for batch, batch_data in enumerate(data_train_loader):

            model.zero_grad()

            train_img = batch_data['image'].cuda()
            train_kp = batch_data['keypoints'].cuda()

            pred_kp = model(train_img)
            pred_kp = pred_kp.view(pred_kp.shape[0], 16, 3)
            train_kp = train_kp.view(train_kp.shape[0], 16, 3)

            loss_value = loss(pred_kp, train_kp)
            batch_train_loss_list.append(loss_value.item())
            loss_value.backward()

            optimizer.step()

            if batch == 0 and not epoch % logging_step:
                print(f'Train batch: {batch}, Train batch loss: {loss_value.data}, Epoch: {epoch}')

        epoch_train_loss_list.append(np.average(batch_train_loss_list))
        test_loss_value = test_lenet(model, data_test_loader)
        epoch_test_loss_list.append(test_loss_value)
        if not len(min_epoch_test_loss_list) or test_loss_value < min_epoch_test_loss_list[-1]:
            min_epoch_test_loss_list.append(test_loss_value)
        else:
            min_epoch_test_loss_list.append(min_epoch_test_loss_list[-1])

        if not epoch % logging_step:
            print(f'Test loss: {test_loss_value}, Epoch: {epoch}')
            plt.figure()
            plt.ylim((0, 800))
            plt.plot(epoch_train_loss_list)
            plt.plot(epoch_test_loss_list, c='orange')
            plt.plot(min_epoch_test_loss_list, c='red')
            plt.show()
            # t = transforms.ToPILImage()
            # plt.figure()
            # show_keypoints(
            #     t(train_img[-1].cpu()),
            #     pred_kp[-1].view(16, 3).squeeze().cpu().detach(),
            #     show_edges=True
            # )
            # fig = plt.figure(figsize=(5, 5))
            # for i in range(4):
            #     sample = data_train[random.randint(0, len(data_train) - 1)]
            #
            #     train_img = sample['image'].cuda()
            #     train_img = train_img.unsqueeze(0)
            #     pred_kp = model(train_img)
            #     pred_kp = pred_kp.view(pred_kp.shape[0], 16, 3)
            #
            #     ax = plt.subplot(2, 2, i + 1)
            #     plt.tight_layout()
            #     ax.set_title('Sample #{}'.format(i))
            #     ax.axis('off')
            #     show_keypoints(
            #         t(train_img[-1].cpu()),
            #         pred_kp[-1].view(16, 3).squeeze().cpu().detach(),
            #         show_edges=i % 2
            #     )
            # plt.show()

        # torch.cuda.empty_cache()
    torch.save(model.state_dict(),
               ('../../models/weights/lenet_3_128_aug_max/LeNet128'
                f'_A{str(alpha).replace(".", "o")}'
                f'_E{epochs}'
                f'_B{train_batch_size}'
                '.weights'))


def train_cpm(model,
              alpha,
              image_size,
              train_batch_size,
              epochs,
              test_batch_size,
              logging_step=5):
    set_random_seed(SEED)

    t = transforms.ToPILImage()

    all_tform = transforms.Compose([
        RandomFlip(0.5, 0.5),
        RandomRatioCrop(0.05, 0.05, 0.95, 0.95),
        RandomRotation((-15, 15)),
    ])

    img_tform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    data_train = AnimalKeypointsDataset(
        json_file_path='../../dataset/cats/train/keypoints_annotations.json',
        image_dir='../../dataset/cats/train/labeled/',
        transform={'all': all_tform,
                   'image': img_tform,
                   'keypoints': transforms.ToTensor()},
        heatmap=True)
    data_train_loader = DataLoader(data_train, batch_size=train_batch_size, shuffle=True, num_workers=0)

    data_test = AnimalKeypointsDataset(
        json_file_path='../../dataset/cats/test/keypoints_annotations.json',
        image_dir='../../dataset/cats/test/labeled/',
        transform={'all': all_tform,
                   'image': img_tform,
                   'keypoints': transforms.ToTensor()},
        heatmap=True)
    data_test_loader = DataLoader(data_test, batch_size=test_batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    # loss = MSECELoss().cuda()
    epoch_train_loss_list = []
    epoch_test_loss_list = []
    min_epoch_test_loss_list = []

    for epoch in range(epochs):
        batch_train_loss_list = []
        for batch, batch_data in enumerate(data_train_loader):

            model.zero_grad()

            train_img = batch_data['image'].cuda()
            train_hmap = batch_data['heatmap'].cuda()

            pred_hmaps_list = model(train_img)

            loss_value = ((pred_hmaps_list[0] - train_hmap) ** 2).sum()  # TODO: move into for loop
            for i in range(1, len(pred_hmaps_list)):
                loss_value += ((pred_hmaps_list[i] - train_hmap) ** 2).sum() / train_batch_size
            batch_train_loss_list.append(loss_value.item())
            loss_value.backward()

            optimizer.step()

            if batch == 0 and not epoch % logging_step:
                print(f'Train batch: {batch}, Train batch loss: {loss_value.data}, Epoch: {epoch}')

        epoch_train_loss_list.append(np.average(batch_train_loss_list))
        test_loss_value = test_cpm(model, data_test_loader, test_batch_size)
        epoch_test_loss_list.append(test_loss_value)
        if not len(min_epoch_test_loss_list) or test_loss_value < min_epoch_test_loss_list[-1]:
            min_epoch_test_loss_list.append(test_loss_value)
        else:
            min_epoch_test_loss_list.append(min_epoch_test_loss_list[-1])

        if not epoch % logging_step:
            print(f'Test loss: {test_loss_value}, Epoch: {epoch}')
            plt.figure()
            # plt.ylim((0, 800))
            plt.plot(epoch_train_loss_list)
            plt.plot(epoch_test_loss_list, c='orange')
            plt.plot(min_epoch_test_loss_list, c='red')
            plt.show()

        if not epoch % 90:
            torch.save(model,
                       ('../../models/weights/cpm_368/entire_model/CPM368'
                        f'_A{str(alpha).replace(".", "o")}'
                        f'_E{epoch + 1525}'
                        f'_B{train_batch_size}'
                        '.pth'))
            # t = transforms.ToPILImage()
            # plt.figure()
            # show_keypoints(
            #     t(train_img[-1].cpu()),
            #     pred_kp[-1].view(16, 3).squeeze().cpu().detach(),
            #     show_edges=True
            # )
            # fig = plt.figure(figsize=(5, 5))
            # for i in range(4):
            #     sample = data_train[random.randint(0, len(data_train) - 1)]
            #
            #     train_img = sample['image'].cuda()
            #     train_img = train_img.unsqueeze(0)
            #     pred_kp = model(train_img)
            #     pred_kp = pred_kp.view(pred_kp.shape[0], 16, 3)
            #
            #     ax = plt.subplot(2, 2, i + 1)
            #     plt.tight_layout()
            #     ax.set_title('Sample #{}'.format(i))
            #     ax.axis('off')
            #     show_keypoints(
            #         t(train_img[-1].cpu()),
            #         pred_kp[-1].view(16, 3).squeeze().cpu().detach(),
            #         show_edges=i % 2
            #     )
            # plt.show()

        # torch.cuda.empty_cache()
    torch.save(model,
               ('../../models/weights/cpm_368/entire_model/CPM368'
                '_aug_min'
                f'_A{str(alpha).replace(".", "o")}'
                f'_E{epochs + 1525}'
                f'_B{train_batch_size}'
                '.pth'))


if __name__ == '__main__':
    INIT_WEIGHT_PATH = '../../models/weights/cpm_368/entire_model/CPM368_A1e-05_E1525_B6.pth'
    ALPHA = 0.00001
    IMAGE_SIZE = (368, 368)
    EPOCHS = 180
    TRAIN_BATCH_SIZE = 6
    TEST_BATCH_SIZE = 4
    LOG_STEP = 30

    model = ConvolutionalPoseMachines(keypoints=16, sub_stages=2).cuda()
    if os.path.isfile(INIT_WEIGHT_PATH):
        model = torch.load(INIT_WEIGHT_PATH)
    else:
        print("Weights not found.")

    train_cpm(
        model,
        ALPHA,
        IMAGE_SIZE,
        TRAIN_BATCH_SIZE,
        EPOCHS,
        TEST_BATCH_SIZE,
        LOG_STEP
    )
