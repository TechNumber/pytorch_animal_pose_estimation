import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from utils.other import hmap_to_keypoints


class LitHMapEstimator(pl.LightningModule):
    def __init__(self, cfg: DictConfig):  # TODO: заменить на класс конфига
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.model.init, cfg=cfg)
        self.loss = instantiate(cfg.loss)
        self.metric = instantiate(cfg.metric)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())
        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return {
            'optimizer': optimizer, 'lr_scheduler': scheduler
        }

    def training_step(self, batch, batch_idx):
        loss, metric = self._calc_loss_metric(batch, batch_idx)
        self.log_dict({'train/loss': loss, 'train/metric': metric})  # TODO: разобраться с логированием
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric = self._calc_loss_metric(batch, batch_idx)
        self.log_dict({'val/loss': loss, 'val/metric': metric})

    def test_step(self, batch, batch_idx):
        loss, metric = self._calc_loss_metric(batch, batch_idx)
        self.log_dict({'test/loss': loss, 'test/metric': metric})

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img = batch['image']
        hmap_pred = self.model(img)
        hmap_pred = hmap_pred[:, -1]
        kp_pred = hmap_to_keypoints(hmap_pred)
        # TODO: добавить логирование изображения вместе с keypoints через torchvision
        return kp_pred

    def _calc_loss_metric(self, batch, batch_idx):
        img, hmap_true = batch['image'], batch['heatmap']
        hmap_pred = self.model(img)
        loss = self.loss(hmap_pred, hmap_true)
        metric = self.metric(hmap_to_keypoints(hmap_pred), hmap_to_keypoints(hmap_true))

        return loss, metric
