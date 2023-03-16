from typing import Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch import nn

from conf.config_dataclasses import Config


class Conv2dBlock(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            ker_size: int,
            pad: int = 0,
            act_layer: Optional[nn.Module] = None,
            pool_layer: Optional[nn.Module] = None
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ker_size, padding=pad)
        self.act = act_layer() if act_layer else nn.Identity()
        self.pool = pool_layer(kernel_size=3, stride=2) if pool_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class ImageFeatureBlock(nn.Module):
    def __init__(
            self,
            n_base_ch: int,
            norm_layer: Optional[nn.Module] = None,
            act_layer: Optional[nn.Module] = None,
            pool_layer: Optional[nn.Module] = None,
            drop_rate: Optional[float] = None
    ):
        super().__init__()
        self.norm = norm_layer(
            (368, 368)) if norm_layer else nn.Identity()  # TODO: check if model runs better without this norm (use albumentations' norm instead)
        self.blocks = nn.ModuleList([
            Conv2dBlock(3, n_base_ch, ker_size=9, pad=4, act_layer=act_layer, pool_layer=pool_layer),
            Conv2dBlock(n_base_ch, n_base_ch, ker_size=9, pad=4, act_layer=act_layer, pool_layer=pool_layer),
            Conv2dBlock(n_base_ch, n_base_ch, ker_size=9, pad=4, act_layer=act_layer, pool_layer=pool_layer)
        ])
        self.drop = nn.Dropout(drop_rate) if drop_rate else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.drop(x)
        return x


class InitialStage(nn.Module):
    def __init__(
            self,
            n_maps: int,
            n_base_ch: int,
            norm_layer: Optional[nn.Module] = None,
            act_layer: Optional[nn.Module] = None,
            pool_layer: Optional[nn.Module] = None,
            drop_rate: Optional[float] = None
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            ImageFeatureBlock(n_base_ch, norm_layer, act_layer, pool_layer, drop_rate),
            Conv2dBlock(n_base_ch, n_base_ch // 4, ker_size=5, pad=2, act_layer=act_layer),
            Conv2dBlock(n_base_ch // 4, n_base_ch * 4, ker_size=9, pad=4, act_layer=act_layer),
            Conv2dBlock(n_base_ch * 4, n_base_ch * 4, ker_size=1, pad=0, act_layer=act_layer),
            Conv2dBlock(n_base_ch * 4, n_maps, ker_size=1, pad=0)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SubsequentStage(nn.Module):
    def __init__(
            self,
            n_maps: int,
            img_feat_ch: int,
            n_base_ch: int,
            norm_layer: Optional[nn.Module] = None,
            act_layer: Optional[nn.Module] = None,
            drop_rate: Optional[float] = None
    ):
        super().__init__()

        self.conv_1 = Conv2dBlock(n_base_ch, img_feat_ch, ker_size=5, pad=2, act_layer=act_layer)

        self.norm = norm_layer((45, 45)) if norm_layer else nn.Identity()
        self.blocks = nn.ModuleList([
            Conv2dBlock(n_maps + img_feat_ch, n_base_ch, ker_size=11, pad=5, act_layer=act_layer),
            Conv2dBlock(n_base_ch, n_base_ch, ker_size=11, pad=5, act_layer=act_layer),
            Conv2dBlock(n_base_ch, n_base_ch, ker_size=11, pad=5, act_layer=act_layer),
            Conv2dBlock(n_base_ch, n_base_ch, ker_size=1, pad=0, act_layer=act_layer),
            Conv2dBlock(n_base_ch, n_maps, ker_size=1, pad=0)
        ])
        self.drop = nn.Dropout(drop_rate) if drop_rate else nn.Identity()

    def forward(self, x: Tensor, img_ref: Tensor) -> Tensor:
        img_ref = self.conv_1(img_ref)
        x = torch.cat((x, img_ref), 1)
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.drop(x)
        return x


class ConvolutionalPoseMachines(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        if 'norm_layer' in cfg.model:
            norm_layer = cfg.model.norm_layer and hydra.utils.get_class(cfg.model.norm_layer)
        if 'act_layer' in cfg.model:
            act_layer = cfg.model.act_layer and hydra.utils.get_class(cfg.model.act_layer)
        if 'pool_layer' in cfg.model:
            pool_layer = cfg.model.pool_layer and hydra.utils.get_class(cfg.model.pool_layer)
        self.init_stage = InitialStage(
            n_maps=cfg.dataset.n_keypoints + cfg.dataset.include_bground_map,
            n_base_ch=cfg.model.n_base_ch,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pool_layer=pool_layer,
            drop_rate=cfg.model.drop_rate
        )

        self.img_feat = ImageFeatureBlock(
            n_base_ch=cfg.model.n_base_ch,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pool_layer=pool_layer,
            drop_rate=cfg.model.drop_rate
        )

        self.subsequent_stages_list = nn.ModuleList(
            [SubsequentStage(
                n_maps=cfg.dataset.n_keypoints + cfg.dataset.include_bground_map,
                img_feat_ch=cfg.model.img_feat_ch,
                n_base_ch=cfg.model.n_base_ch,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop_rate=cfg.model.drop_rate
            ) for i in range(cfg.model.n_substages)]
        )

    def forward(self, x: Tensor) -> Tensor:
        img_ref = self.img_feat(x)
        outputs = [self.init_stage(x)]

        for sub_stage in self.subsequent_stages_list:
            outputs.append(sub_stage(outputs[-1], img_ref))

        return torch.stack(outputs, dim=0)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input = torch.rand((5, 3, 368, 368), device=device)
    print('Input shape:', input.shape)

    cfg = DictConfig({
        'model': {
            'n_substages': 3,
            'n_base_ch': 128,
            'img_feat_ch': 32,
            'act_layer': 'torch.nn.GELU',
            'pool_layer': 'torch.nn.MaxPool2d'
        },
        'dataset': {
            'n_keypoints': 16,
            'include_bground_map': False
        }
    })

    with torch.inference_mode():
        model = ConvolutionalPoseMachines(cfg).to(device)
        output = model(input)
        print('Output shape:', output.shape)
    for c in model.children():
        print(c)
