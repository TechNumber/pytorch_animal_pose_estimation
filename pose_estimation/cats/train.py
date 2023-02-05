import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_loading.animal_keypoints_dataset import AnimalKeypointsDataset
from models.conv_pose_machines import ConvolutionalPoseMachines
from pose_estimation.cats.test import test
from utils.set_random_seed import set_random_seed, SEED
from utils.losses import HMapsMSELoss
from utils.model_saver import ModelSaver
from utils.transforms import RandomRotation, RandomFlip, RandomRatioCrop


def train(model,
          data_train,
          data_test,
          loss,
          optimizer,
          epochs,
          model_saver=None,
          logging_step=5,
          device=torch.device('cpu')):
    t = transforms.ToPILImage()

    epoch_train_loss_list = []
    epoch_test_loss_list = []
    min_epoch_test_loss_list = []

    for epoch in range(epochs):
        batch_train_loss_list = []
        for batch, batch_data in enumerate(data_train):
            model.zero_grad()

            train_img = batch_data['image'].to(device)
            train_hmap = batch_data['heatmap'].to(device)

            pred_hmaps = model(train_img)
            loss_value = loss(pred_hmaps, train_hmap.unsqueeze(1))

            batch_train_loss_list.append(loss_value.item())
            loss_value.backward()

            optimizer.step()

        epoch_train_loss_list.append(np.average(batch_train_loss_list))
        test_loss_value = test(model, loss, data_test, device)
        epoch_test_loss_list.append(test_loss_value)
        if not len(min_epoch_test_loss_list) or test_loss_value < min_epoch_test_loss_list[-1]:
            min_epoch_test_loss_list.append(test_loss_value)
        else:
            min_epoch_test_loss_list.append(min_epoch_test_loss_list[-1])

        if not epoch % logging_step:
            print(f'Train loss: {epoch_train_loss_list[-1]}, Epoch: {epoch}')
            print(f'Test loss: {test_loss_value}, Epoch: {epoch}')
            plt.figure()
            plt.ylim((0, 10))
            plt.plot(epoch_train_loss_list)
            plt.plot(epoch_test_loss_list, c='orange')
            plt.plot(min_epoch_test_loss_list, c='red')
            plt.show()

        model_saver.save(model, epoch)

    model_saver.save(model, epochs)


if __name__ == '__main__':
    set_random_seed(SEED)

    INIT_WEIGHT_PATH = '../../models/weights/ConvolutionalPoseMachines_3_stages_(entire_model)/entire_model/v2/CPM368_aug_min_A1e-057_E20_B6.pth'
    ALPHA = 0.00001
    IMAGE_SIZE = (368, 368)
    EPOCHS = 1800
    TRAIN_BATCH_SIZE = 6
    TEST_BATCH_SIZE = 4
    LOG_STEP = 30
    SAVE_MODEL_STEP = 90
    START_EPOCH = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    all_tform = transforms.Compose([
        RandomFlip(0.5, 0.5),
        RandomRatioCrop(0.1, 0.1, 0.9, 0.9),
        RandomRotation((-30, 30)),
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
                   'keypoints': transforms.ToTensor()},
        heatmap=True)
    data_train_loader = DataLoader(data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)

    data_test = AnimalKeypointsDataset(
        json_file_path='../../dataset/cats/test/keypoints_annotations.json',
        image_dir='../../dataset/cats/test/labeled/',
        transform={'all': all_tform,
                   'image': img_tform,
                   'keypoints': transforms.ToTensor()},
        heatmap=True)
    data_test_loader = DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=0)

    model = ConvolutionalPoseMachines(
        n_keypoints=16,
        n_substages=2,
        n_base_ch=64,
        img_feat_ch=16,
        device=device
    )
    if os.path.isfile(INIT_WEIGHT_PATH):
        model.load_state_dict(torch.load(INIT_WEIGHT_PATH))
    else:
        print("Weights not found.")
    optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)
    loss = HMapsMSELoss().to(device)

    model_saver = ModelSaver(model,
                             TRAIN_BATCH_SIZE,
                             save_freq=SAVE_MODEL_STEP,
                             start_epoch=START_EPOCH,
                             loss=loss,
                             optimizer=optimizer)

    train(
        model=model,
        data_train=data_train_loader,
        data_test=data_test_loader,
        loss=loss,
        optimizer=optimizer,
        epochs=EPOCHS,
        model_saver=model_saver,
        logging_step=LOG_STEP,
        device=device
    )
