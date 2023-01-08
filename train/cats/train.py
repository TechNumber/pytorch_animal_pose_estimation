import torch
from data_loading.data_loading import AnimalKeypointsDataset
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
        pred = pred.view(pred.shape[0], 16, 3)
        true = true.view(true.shape[0], 16, 3)
        # TODO: move input transformation to train()
        return torch.nn.functional.mse_loss(pred[:, :, :-1], true[:, :, :-1]) + \
               torch.nn.functional.cross_entropy(pred[:, :, -1], true[:, :, -1])


def train():
    INIT_WEIGHT_PATH = './weights/simplepose9.weights'
    ALPHA = 0.0001
    IMAGE_SIZE = (128, 128)
    EPOCHS = 2
    BATCH_SIZE = 10

    set_random_seed(SEED)

    tform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    data_train = AnimalKeypointsDataset(json_file_path='./dataset/keypoints_annotations.json',
                                        image_dir='./dataset/labeled/',
                                        transform={'image': tform,
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

            loss = loss(pred_kp, train_kp)
            loss.backward()
            optimizer.step()

            if batch % 2:
                print(f'Batch: {batch}, Loss: {loss.data}, Epoch: {epoch}')

        t = transforms.ToPILImage()
        show_keypoints(
            t(train_img[-1].cpu()),
            pred_kp[-1].view(16, 3).squeeze().cpu().detach(),
            show_edges=True
        )

        # torch.cuda.empty_cache()
    torch.save(model.state_dict(),
               ('./weights/LeNet128'
                f'_A{str(ALPHA).replace(".", "o")}'
                f'_E{EPOCHS}'
                f'_B{BATCH_SIZE}'
                '.weights'))
