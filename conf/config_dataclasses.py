from dataclasses import dataclass, field
from typing import Optional, List, Any, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING


@dataclass
class LitHMapEstimatorConfig:
    _target_: str = 'models.lightning_hmap_estimator.LitHMapEstimator'
    _recursive_: bool = False


@dataclass
class CPMConfig:
    init: dict = field(
        default_factory=lambda: {
            '_target_': 'models.conv_pose_machines.ConvolutionalPoseMachines',
            '_recursive_': False
        }
    )
    norm_layer: Optional[str] = None
    act_layer: str = 'torch.nn.GELU'
    pool_layer: str = 'torch.nn.MaxPool2d'
    drop_rate: float = 0.
    n_substages: int = 3
    n_base_ch: int = 128
    img_feat_ch: int = 32


@dataclass
class AKDConfig:
    init: dict = field(
        default_factory=lambda: {
            '_target_': 'datasets.cats_datamodule.AKDDataModule',
            '_recursive_': False
        }
    )
    data_dir: str = './data'
    heatmap_size: int = 45
    produce_visibility: bool = True
    subset_size: int = 1500
    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 4
    n_keypoints: int = 16
    include_bground_map: bool = False


@dataclass
class MSEConfig:
    _target_: str = 'torch.nn.MSELoss'
    reduction: Optional[str] = 'sum'


@dataclass
class PCKConfig:
    _target_: str = 'utils.metrics.PCK'
    thr: float = 0.2


@dataclass
class AdamConfig:
    _target_: str = 'torch.optim.Adam'
    lr: float = 1e-5


@dataclass
class AdamWConfig:
    _target_: str = 'torch.optim.AdamW'
    lr: float = 1e-5  # TODO: ${training.lr}


@dataclass
class OneCycleConfig:
    _target_: str = 'torch.optim.lr_scheduler.OneCycleLR'
    max_lr: float = 1e-5
    total_steps: Optional[int] = None


@dataclass
class TrainerConfig:
    max_epochs: int = 20
    accumulate_grad_batches: Optional[int] = None
    accelerator: str = 'auto'
    deterministic: bool = False


@dataclass
class RandomFlipConfig:
    @dataclass
    class RandomFlipParams:
        _target_: str = 'utils.transforms.RandomFlip'
        ver_cv: float = 0.5
        hor_cv: float = 0.5

    flip: RandomFlipParams = field(default=RandomFlipParams)


@dataclass
class RandomRatioCropConfig:
    @dataclass
    class RandomRatioCropParams:
        _target_: str = 'utils.transforms.RandomRatioCrop'
        max_top_offset: float = 0.1
        max_left_offset: float = 0.1
        min_crop_height: float = 0.9
        min_crop_width: float = 0.9

    ratio_crop: RandomRatioCropParams = field(default=RandomRatioCropParams)


@dataclass
class RandomRotationConfig:
    @dataclass
    class RandomRotationParams:
        _target_: str = 'utils.transforms.RandomRotation'
        degrees: List[int] = field(default_factory=lambda: [-30, 30])

    rotation: RandomRotationParams = field(default=RandomRotationParams)


@dataclass
class ResizeConfig:
    @dataclass
    class ResizeParams:
        _target_: str = 'torchvision.transforms.Resize'
        size: List[int] = field(default_factory=lambda: [368, 368])

    resize: ResizeParams = field(default=ResizeParams)


@dataclass
class ToTensorConfig:
    @dataclass
    class ToTensorParams:
        _target_: str = 'torchvision.transforms.ToTensor'

    to_tensor: ToTensorParams = field(default=ToTensorParams)


@dataclass
class ModelCheckpointConfig:
    @dataclass
    class ModelCheckpointParams:
        _target_: str = 'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'
        dirpath: str = 'checkpoints'
        save_top_k: int = 2
        monitor: str = 'val/loss'
        mode: str = 'min'

    model_checkpoint: ModelCheckpointParams = field(default=ModelCheckpointParams)


@dataclass
class LRMonitorConfig:
    @dataclass
    class LRMonitorParams:
        _target_: str = 'pytorch_lightning.callbacks.LearningRateMonitor'

    lr_monitor: LRMonitorParams = field(default=LRMonitorParams)


@dataclass
class EarlyStoppingConfig:
    @dataclass
    class EarlyStoppingParams:
        _target_: str = 'pytorch_lightning.callbacks.EarlyStopping'
        monitor: str = 'val/loss'
        mode: str = 'min'
        patience: int = 10
        min_delta: float = 0.001  # TODO: adjust
        verbose: bool = True

    early_stopping: EarlyStoppingParams = field(default=EarlyStoppingParams)


@dataclass
class WandbLoggerConfig:
    _target_: str = 'pytorch_lightning.loggers.wandb.WandbLogger'
    project: str = 'animal_pose_estimation'
    name: Optional[str] = None
    id: Optional[str] = None
    resume: Optional[Union[bool, str]] = None


cs = ConfigStore.instance()
cs.store(group='lit_module', name='hmap_estimator', node=LitHMapEstimatorConfig)
cs.store(group='model', name='cpm', node=CPMConfig)
cs.store(group='dataset', name='akd', node=AKDConfig)
cs.store(group='loss', name='mse', node=MSEConfig)
cs.store(group='metric', name='pck', node=PCKConfig)
cs.store(group='optimizer', name='adam', node=AdamConfig)
cs.store(group='optimizer', name='adamw', node=AdamWConfig)
cs.store(group='scheduler', name='onecycle', node=OneCycleConfig)
cs.store(group='trainer', name='default_trainer', node=TrainerConfig)

cs.store(group='train_augmentations.all', name='flip', node=OmegaConf.to_yaml(RandomFlipConfig))
cs.store(group='train_augmentations.all', name='ratio_crop', node=OmegaConf.to_yaml(RandomRatioCropConfig))
cs.store(group='train_augmentations.all', name='rotation', node=OmegaConf.to_yaml(RandomRotationConfig))
cs.store(group='train_augmentations.image', name='resize', node=OmegaConf.to_yaml(ResizeConfig))
cs.store(group='train_augmentations.image', name='to_tensor', node=OmegaConf.to_yaml(ToTensorConfig))

cs.store(group='test_augmentations.all', name='flip', node=OmegaConf.to_yaml(RandomFlipConfig))
cs.store(group='test_augmentations.all', name='ratio_crop', node=OmegaConf.to_yaml(RandomRatioCropConfig))
cs.store(group='test_augmentations.all', name='rotation', node=OmegaConf.to_yaml(RandomRotationConfig))
cs.store(group='test_augmentations.image', name='resize', node=OmegaConf.to_yaml(ResizeConfig))
cs.store(group='test_augmentations.image', name='to_tensor', node=OmegaConf.to_yaml(ToTensorConfig))

cs.store(group='callbacks', name='model_checkpoint', node=OmegaConf.to_yaml(ModelCheckpointConfig))
cs.store(group='callbacks', name='lr_monitor', node=OmegaConf.to_yaml(LRMonitorConfig))
cs.store(group='callbacks', name='early_stopping', node=OmegaConf.to_yaml(EarlyStoppingConfig))
cs.store(group='logger', name='wandb', node=WandbLoggerConfig)

defaults = [
    {'lit_module': 'hmap_estimator'},
    {'model': 'cpm'},
    {'loss': 'mse'},
    {'metric': 'pck'},
    {'optimizer': 'adam'},
    {'scheduler': 'onecycle'},
    {'dataset': 'akd'},
    {'trainer': 'default_trainer'},
    {'logger': 'wandb'},
    '_self_',
]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    lit_module: Any = MISSING
    model: Any = MISSING
    loss: Any = MISSING
    metric: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING
    dataset: Any = MISSING

    @dataclass
    class AugmentationsVars:
        all: Optional[Any] = None
        image: Optional[Any] = None
        target: Optional[Any] = None

    train_augmentations: AugmentationsVars = field(default=AugmentationsVars)
    test_augmentations: AugmentationsVars = field(default=AugmentationsVars)

    trainer: Any = MISSING
    callbacks: Optional[Any] = None
    logger: Any = MISSING

    seed: int = 37


cs.store(name='base_config', node=Config)


@hydra.main(version_base=None, config_name='config', config_path='./')
def read_config(cfg: Config):
    print(OmegaConf.to_yaml(cfg))



if __name__ == '__main__':
    read_config()
