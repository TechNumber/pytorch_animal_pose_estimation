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
from pose_estimation.cats.test import test


def train():
    INIT_WEIGHT_PATH = '../../models/weights/lenet_3_128_aug_max/LeNet128_A0o0001_E2500_B200.weights'
    ALPHA = 0.00001
    IMAGE_SIZE = (128, 128)
    EPOCHS = 2000
    BATCH_SIZE = 200

    set_random_seed(SEED)

    t = transforms.ToPILImage()

    all_tform = transforms.Compose([
        RandomFlip(0.5, 0.5),
        RandomRatioCrop(0.15, 0.15, 0.85, 0.85),
        RandomRotation((-45, 45)),
    ])

    img_tform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    data_train = AnimalKeypointsDataset(
        json_file_path='../../dataset/cats/train/keypoints_annotations.json',
        image_dir='../../dataset/cats/train/labeled/',
        transform={'all': all_tform,
                   'image': img_tform,
                   'keypoints': transforms.ToTensor()})
    data_train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    data_test = AnimalKeypointsDataset(
        json_file_path='../../dataset/cats/test/keypoints_annotations.json',
        image_dir='../../dataset/cats/test/labeled/',
        transform={'all': all_tform,
                   'image': img_tform,
                   'keypoints': transforms.ToTensor()})
    data_test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = LeNet128().cuda()
    if os.path.isfile(INIT_WEIGHT_PATH):
        model.load_state_dict(torch.load(INIT_WEIGHT_PATH))
    else:
        print("Weights not found.")

    optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)
    loss = MSECELoss().cuda()
    epoch_train_loss_list = []
    epoch_test_loss_list = []
    min_epoch_test_loss_list = []

    for epoch in range(EPOCHS):
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

            if batch==0 and not epoch % 50:
                print(f'Train batch: {batch}, Train batch loss: {loss_value.data}, Epoch: {epoch}')

        epoch_train_loss_list.append(np.average(batch_train_loss_list))
        test_loss_value = test(model, data_test_loader)
        epoch_test_loss_list.append(test_loss_value)
        if not len(min_epoch_test_loss_list) or test_loss_value < min_epoch_test_loss_list[-1]:
            min_epoch_test_loss_list.append(test_loss_value)
        else:
            min_epoch_test_loss_list.append(min_epoch_test_loss_list[-1])

        if not epoch % 50:
            print(f'Test loss: {test_loss_value}, Epoch: {epoch}')
            plt.figure()
            plt.ylim((0, 150))
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
                f'_A{str(ALPHA).replace(".", "o")}'
                f'_E{EPOCHS}'
                f'_B{BATCH_SIZE}'
                '.weights'))


if __name__ == '__main__':
    train()
