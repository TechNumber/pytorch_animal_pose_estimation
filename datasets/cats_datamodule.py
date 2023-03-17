import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Subset, DataLoader

from conf.config_dataclasses import Config
from datasets.animal_keypoints_dataset import AKD
from torchvision import transforms


class AKDDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.train_all_transform = cfg.train_augmentations.all and transforms.Compose(list(
            instantiate(cfg.train_augmentations.all).values()))
        self.train_image_transform = cfg.train_augmentations.image and transforms.Compose(list(
            instantiate(cfg.train_augmentations.image).values()))
        self.train_target_transform = cfg.train_augmentations.target and transforms.Compose(list(
            instantiate(cfg.train_augmentations.target).values()))

        self.test_all_transform = cfg.test_augmentations.all and transforms.Compose(list(
            instantiate(cfg.test_augmentations.all).values()))
        self.test_image_transform = cfg.test_augmentations.image and transforms.Compose(list(
            instantiate(cfg.test_augmentations.image).values()))
        self.test_target_transform = cfg.test_augmentations.target and transforms.Compose(list(
            instantiate(cfg.test_augmentations.target).values()))

    def prepare_data(self):
        AKD(split='train', root=self.cfg.dataset.data_dir, download=True)
        AKD(split='test', root=self.cfg.dataset.data_dir, download=True)

    def setup(self, stage: str):
        kwargs = {
            'root': self.cfg.dataset.data_dir,
            'all_transform': self.train_all_transform,
            'image_transform': self.train_image_transform,
            'target_transform': self.train_target_transform,
            'heatmap_size': self.cfg.dataset.heatmap_size if 'heatmap_size' in self.cfg.dataset else None,
            'produce_visibility': self.cfg.dataset.produce_visibility
        }

        if stage == 'fit':
            self.akd_train = AKD(**kwargs, split='train')
            self.akd_val = AKD(**kwargs, split='test')
            if 'subset_size' in self.cfg.dataset:
                if self.cfg.dataset.subset_size < len(self.akd_train):
                    self.akd_train = Subset(self.akd_train, torch.arange(self.cfg.dataset.subset_size))
                if self.cfg.dataset.subset_size < len(self.akd_val):
                    self.akd_val = Subset(self.akd_val, torch.arange(self.cfg.dataset.subset_size))

        if stage == 'val':
            self.akd_val = AKD(**kwargs, split='test')
            if 'subset_size' in self.cfg.dataset and self.cfg.dataset.subset_size < len(self.akd_val):
                self.akd_val = Subset(self.akd_val, torch.arange(self.cfg.dataset.subset_size))

        if stage == 'test':
            self.akd_test = AKD(
                root=self.cfg.dataset.data_dir,
                split='test',
                all_transform=self.test_all_transform,
                image_transform=self.test_image_transform,
                target_transform=self.test_target_transform,
                heatmap_size = self.cfg.dataset.heatmap_size if 'heatmap_size' in self.cfg.dataset else None,
                produce_visibility=self.cfg.dataset.produce_visibility
            )
            if 'subset_size' in self.cfg.dataset and self.cfg.dataset.subset_size < len(self.akd_test):
                self.akd_test = Subset(self.akd_test, torch.arange(self.cfg.dataset.subset_size))

        if stage == 'predict':
            self.akd_predict = AKD(
                root=self.cfg.dataset.data_dir,
                split='test',
                all_transform=self.test_all_transform,
                image_transform=self.test_image_transform,
                target_transform=self.test_target_transform,
                # produce_visibility=self.cfg.dataset.produce_visibility
            )
            if 'subset_size' in self.cfg.dataset and self.cfg.dataset.subset_size < len(self.akd_predict):
                self.akd_predict = Subset(self.akd_predict, torch.arange(self.cfg.dataset.subset_size))

    def train_dataloader(self):
        return DataLoader(
            self.akd_train,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=self.cfg.dataset.shuffle,
            num_workers=self.cfg.dataset.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.akd_val,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.akd_test,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.akd_predict,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers
        )
