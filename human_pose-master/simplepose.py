import deeplake

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

from models.simplepose import SimplePose
from models.datasets.lsp import LSPet, LSP

import numpy as np
from vis import show_pose

from torch.utils.tensorboard import SummaryWriter

import random
from pytorch_lightning import seed_everything


def set_random_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    seed_everything(s, workers=True)


SEED = 17

# Load data
batch_size = 10  # Кол-во записей в пакете, передаваемом нейросети за раз
image_size = (128, 128)  # Размер входного изображения
# hmap_size = 32

tform = transforms.Compose([  # Объявление трансформации для исходных изображений:
    transforms.ToPILImage(),
    transforms.Resize(image_size),  # Рескейл изображений до заданного размера
    transforms.ToTensor(),  # Приведения исходного изображения к формату тензора
    # transforms.Normalize([0.5], [0.5]),
])

# lsp = LSPet('./models/datasets/lspet_dataset/', transform, image_size)
# lsp_loader = DataLoader(lsp, batch_size, shuffle=True, num_workers=1)

dl_train = deeplake.load("hub://activeloop/lsp-train")  # Получение данных
# Создание объекта, позволяющего итерировать данные
lsp_train_loader = dl_train.pytorch(
    tensors=["images", "keypoints"],
    decode_method={'images': 'numpy'},
    transform={'images': tform, 'keypoints': None},
    batch_size=batch_size, shuffle=True, num_workers=3
)


class MSELoss(torch.nn.Module):  # Лосс-функция средний квадрат ошибок

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, joints):
        output = output.view(output.shape[0], 14, 2)
        joints = joints[:, :, :-1]
        return torch.sum((output - joints) ** 2)


# Initialize the model
set_random_seed(SEED)
model = SimplePose().cuda()  # Инициализация модели и её выгрузка на ГПУ
model.load_state_dict(torch.load('./weights/simplepose9.weights'))  # Загрузка pre-trained weights
criterion = MSELoss().cuda()  # Инициализация объекта лосс-функции и его выгрузка на ГПУ
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Инициализация оптимизатора

# # Add network into tensorboard
# data = next(iter(lsp_train_loader))
#
# # Создание тензорборда
# tb = SummaryWriter()
# tb.add_graph(model, data[0].cuda())
# # tb.close()

# Train the model
for epoch in range(1):
    for batch, batch_data in enumerate(lsp_train_loader):

        model.zero_grad()

        input_image = batch_data['images'].cuda()
        joints_hmap = batch_data['keypoints'].cuda()
        # mask = batch_data[3].cuda()

        prediction = model(input_image)

        loss = criterion(prediction, joints_hmap)
        loss.backward()
        optimizer.step()

        # # Tensorboard
        # tb.add_scalar('Loss', loss, 1250*epoch + batch)

        if batch % 10 == 0:
            print('Batch: {}, Loss: {}, Epoch: {}'.format(batch, loss.data, epoch))
            # Add keypoints loss of testing data

    # # Tensorboard
    # for name, weight in model.named_parameters():
    #     tb.add_histogram(name, weight, epoch)
    #     tb.add_histogram(f'{name}.grad', weight.grad, epoch)

    torch.cuda.empty_cache()
    torch.save(model.state_dict(), './weights/simplepose{}.weights'.format(epoch))

# tb.close()

# lsp_test = LSP('./dataset/lsp_dataset/', transform, image_size)

dl_test = deeplake.load("hub://activeloop/lsp-test")
lsp_test_loader = dl_test.pytorch(
    tensors=["images", "keypoints"],
    decode_method={'images': 'numpy'},
    transform={'images': tform, 'keypoints': None},
    batch_size=batch_size, shuffle=True, num_workers=3
)
#
# test_image, *_ = lsp_test_loader.__getitem__(np.random.randint(len(lsp_test_loader)))
# test_image, *_ = lsp_test_loader[np.random.randint(len(lsp_test_loader))]
test_image, *_ = next(iter(lsp_test_loader))
test_predictions = model(test_image[0].cuda())
show_pose(test_image[0], test_predictions.squeeze().cpu().detach())
