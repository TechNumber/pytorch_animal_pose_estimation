import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict, OmegaConf
from torchvision.transforms import functional
from torchvision.utils import draw_keypoints, make_grid

from conf.config_dataclasses import Config
from utils.other import hmap_to_keypoints
from utils.visualization import show_keypoints


class LitHMapEstimator(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.cfg_container = OmegaConf.to_container(cfg, resolve=True)
        self.model = instantiate(cfg.model.init, cfg=cfg)
        # for c in self.model.children():
        #     print(c)
        self.loss = instantiate(cfg.loss)
        self.metric = instantiate(cfg.metric)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())
        if 'total_steps' in self.cfg.scheduler:
            self.cfg.scheduler.total_steps = self.trainer.estimated_stepping_batches
        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return {
            'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def training_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
            wandb.watch(self.model, log='all', log_freq=100)

        loss, metric = self._calc_loss_metric(batch, batch_idx)
        self.log_dict({'train/loss': loss, 'train/metric': metric}, on_step=False, on_epoch=True)  # TODO: разобраться с логированием
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric = self._calc_loss_metric(batch, batch_idx)
        self.log_dict({'val/loss': loss, 'val/metric': metric}, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        img, hmap_true = batch['image'], batch['heatmap']
        hmap_pred = self.model(img)
        kp_pred = hmap_to_keypoints(hmap_pred[-1])
        kp_true = hmap_to_keypoints(hmap_true)
        loss = self.loss(hmap_pred, hmap_true)
        metric = self.metric(kp_pred, kp_true)

        self.logger.log_image('predicted_keypoints', show_keypoints(kp_pred, img))
        self.log_dict({'test/loss': loss, 'test/metric': metric}, on_step=False, on_epoch=True)
        # self.experiment.logger.log({'test/loss': loss, 'test/metric': metric, 'test/predicted_keypoints': wandb.Image(img)})

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img = batch['image']
        hmap_pred = self.model(img)
        kp_pred = hmap_to_keypoints(hmap_pred[-1])

        return kp_pred, show_keypoints(kp_pred, img)

    def _calc_loss_metric(self, batch, batch_idx):
        img, hmap_true = batch['image'], batch['heatmap']
        hmap_pred = self.model(img)
        loss = self.loss(hmap_pred, hmap_true)
        metric = self.metric(hmap_to_keypoints(hmap_pred[-1]), hmap_to_keypoints(hmap_true))

        return loss, metric
