import random

import deeplake
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
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
    transforms.ToTensor(),  # Приведение исходного изображения к формату тензора
    # transforms.Normalize([0.5], [0.5]),
])

dl_train = deeplake.load("hub://activeloop/lsp-train")  # Получение данных
# Создание объекта, позволяющего итерировать данные

lsp_train_loader = dl_train.pytorch(
    tensors=["images", "keypoints"],
    decode_method={'images': 'numpy'},
    transform={'images': tform, 'keypoints': None},
    batch_size=batch_size, shuffle=False, num_workers=3
)

data = next(iter(lsp_train_loader))

image, keypoints = data
