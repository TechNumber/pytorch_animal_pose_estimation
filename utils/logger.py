import os.path
from typing import Union, Optional

import wandb
from torch import nn
from torch import optim


class Logger:
    def __init__(
            self,
            model: nn.Module,
            img_size: Union[int, list[int, int], tuple[int, int]],
            epochs: int,
            train_batch_size: int,
            loss: Optional[nn.Module] = None,
            optimizer: Optional[optim.Optimizer] = None,
            **kwargs
    ) -> None:
        model_name = str(type(model))
        model_name = model_name[model_name.rfind('.') + 1:-2]

        self.config = {
            'model': model_name,
            'img_size': (img_size, img_size) if isinstance(img_size, int) else img_size,
            'epochs': epochs,
            'train_batch_size': train_batch_size,
            **kwargs
        }

        if loss:
            loss_name = str(type(loss))
            loss_name = loss_name[loss_name.rfind('.') + 1:-2]
            self.config['loss'] = loss_name

        if optimizer:
            opt_name = str(type(optimizer))
            opt_name = opt_name[opt_name.rfind('.') + 1:-2]
            self.config['optimizer'] = opt_name
            opt_params = optimizer.state_dict()['param_groups'][0]
            if 'lr' in opt_params.keys():
                self.config['learning_rate'] = opt_params['lr']
            self.config['optimizer_params'] = {
                k: v for k, v in opt_params.items() if k not in ['params', 'lr'] and v is not None
            }

        if not os.path.exists('./wandb_runs'):
            os.mkdir('./wandb_runs')
        # TODO: .env
        self.run = wandb.init(
            dir='./wandb_runs',
            project="animal_pose_estimation",
            anonymous='must',
            config=self.config
        )

    def watch(self, model: nn.Module):
        self.run.watch(model)

    def log(self, metrics: dict):
        self.run.log(metrics)

    def finish(self):
        self.run.finish()


if __name__ == "__main__":
    import torch
    from models.conv_pose_machines import ConvolutionalPoseMachines
    from utils.set_random_seed import set_random_seed, SEED
    from utils.losses import HMapsMSELoss

    set_random_seed(SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    INIT_WEIGHT_PATH = '../../models/weights/ConvolutionalPoseMachines_4_stages/HMapsMSELoss/Adam_lr_1e-05_betas_(0o9_0o999)_eps_1e-08/ConvolutionalPoseMachines_E899_B5.pth'
    ALPHA = 0.00001
    IMAGE_SIZE = (368, 368)
    N_SUBSTAGES = 2
    EPOCHS = 900
    TRAIN_BATCH_SIZE = 5
    TEST_BATCH_SIZE = 5
    LOG_STEP = 30
    SAVE_MODEL_STEP = 90
    START_EPOCH = 900

    model = ConvolutionalPoseMachines(
        n_keypoints=16,
        n_substages=N_SUBSTAGES
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)
    loss = HMapsMSELoss().to(device)

    logger = Logger(
        model,
        IMAGE_SIZE,
        EPOCHS,
        TRAIN_BATCH_SIZE,
        loss,
        optimizer,
        n_substages=N_SUBSTAGES,
        dataset='cats'
    )

    logger.log({'train/loss': 1.4, 'train/acc': 0.6})
    logger.log({'test/loss': 1.7, 'test/acc': 0.56})

    logger.log({'train/loss': 1.25, 'train/acc': 0.7})
    logger.log({'test/loss': 1.53, 'test/acc': 0.61})

    logger.log({'train/loss': 1.17, 'train/acc': 0.8})
    logger.log({'test/loss': 1.26, 'test/acc': 0.72})

    logger.finish()
