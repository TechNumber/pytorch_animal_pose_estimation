import matplotlib.pyplot as plt
import torch
from data_loading.animal_keypoints_dataset import AnimalKeypointsDataset, RandomRotation, RandomFlip, RandomRatioCrop
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from models.lenet_128 import LeNet128
from set_random_seed import set_random_seed, SEED
from visualization.keypoints import show_keypoints


class MSECELoss(torch.nn.Module):

    def __init__(self):
        super(MSECELoss, self).__init__()

    def forward(self, pred, true):
        # pred = pred.view(pred.shape[0], 16, 3)
        # true = true.view(true.shape[0], 16, 3)
        return torch.nn.functional.mse_loss(pred[:, :, :-1], true[:, :, :-1]) + \
               torch.nn.functional.cross_entropy(pred[:, :, -1], true[:, :, -1])


def train():
    INIT_WEIGHT_PATH = '../../models/weights/simplepose9.weights'
    ALPHA = 0.0001
    IMAGE_SIZE = (128, 128)
    EPOCHS = 3000
    BATCH_SIZE = 30

    set_random_seed(SEED)

    t = transforms.ToPILImage()

    all_tform = transforms.Compose([
        RandomFlip(0.5, 0.5),
        RandomRatioCrop(0.1, 0.1, 0.9, 0.9),
        RandomRotation((-45, 45)),
    ])

    img_tform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    data_train = AnimalKeypointsDataset(
        json_file_path='../../dataset/cats/keypoints_annotations.json',
        image_dir='../../dataset/cats/labeled/',
        transform={'all': all_tform,
                   'image': img_tform,
                   'keypoints': transforms.ToTensor()})
    data_train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = LeNet128().cuda()
    if os.path.isfile(INIT_WEIGHT_PATH):
        model.load_state_dict(torch.load(INIT_WEIGHT_PATH))

    loss = MSECELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)

    for epoch in range(EPOCHS):
        for batch, batch_data in enumerate(data_train_loader):

            model.zero_grad()

            train_img = batch_data['image'].cuda()
            train_kp = batch_data['keypoints'].cuda()

            pred_kp = model(train_img)
            pred_kp = pred_kp.view(pred_kp.shape[0], 16, 3)
            train_kp = train_kp.view(train_kp.shape[0], 16, 3)

            loss_value = loss(pred_kp, train_kp)
            loss_value.backward()
            optimizer.step()

            if batch % 2:
                print(f'Batch: {batch}, Loss: {loss_value.data}, Epoch: {epoch}')

        if epoch % 60 == 0:
            # t = transforms.ToPILImage()
            # plt.figure()
            # show_keypoints(
            #     t(train_img[-1].cpu()),
            #     pred_kp[-1].view(16, 3).squeeze().cpu().detach(),
            #     show_edges=True
            # )
            fig = plt.figure(figsize=(5, 5))
            for i in range(4):
                sample = data_train[i]

                train_img = sample['image'].cuda()
                train_img = train_img.unsqueeze(0)
                pred_kp = model(train_img)
                pred_kp = pred_kp.view(pred_kp.shape[0], 16, 3)

                ax = plt.subplot(2, 2, i + 1)
                plt.tight_layout()
                ax.set_title('Sample #{}'.format(i))
                ax.axis('off')
                show_keypoints(
                    t(train_img[-1].cpu()),
                    pred_kp[-1].view(16, 3).squeeze().cpu().detach(),
                    show_edges=i % 2
                )
            plt.show()

        # torch.cuda.empty_cache()
    torch.save(model.state_dict(),
               ('../../models/weights/LeNet128'
                f'_A{str(ALPHA).replace(".", "o")}'
                f'_E{EPOCHS}'
                f'_B{BATCH_SIZE}'
                '.weights'))


if __name__ == '__main__':
    train()