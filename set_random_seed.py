import random
import numpy as np
import torch
from pytorch_lightning import seed_everything


def set_random_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    seed_everything(s, workers=True)


SEED = 27
