from typing import Optional

import torch
from torch import Tensor
from torch import nn


class Conv2dBlock(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            ker_size: int,
            pad: int = 0,
            act: bool = False,
            pool: bool = False
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ker_size, padding=pad)
        self.act = nn.GELU() if act else None
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2) if pool else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.act:
            x = self.act(x)
        if self.pool:
            x = self.pool(x)
        return x


class ImageFeatureBlock(nn.Module):
    def __init__(self, n_base_ch: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv2dBlock(3, n_base_ch, ker_size=9, pad=4, act=True, pool=True),
            Conv2dBlock(n_base_ch, n_base_ch, ker_size=9, pad=4, act=True, pool=True),
            Conv2dBlock(n_base_ch, n_base_ch, ker_size=9, pad=4, act=True, pool=True)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class InitialStage(nn.Module):
    def __init__(self, n_maps: int, n_base_ch: int):
        super().__init__()

        self.blocks = nn.ModuleList([
            ImageFeatureBlock(n_base_ch),
            Conv2dBlock(n_base_ch, n_base_ch // 4, ker_size=5, pad=2, act=True, pool=False),
            Conv2dBlock(n_base_ch // 4, n_base_ch * 4, ker_size=9, pad=4, act=True, pool=False),
            Conv2dBlock(n_base_ch * 4, n_base_ch * 4, ker_size=1, pad=0, act=True, pool=False),
            Conv2dBlock(n_base_ch * 4, n_maps, ker_size=1, pad=0, act=False, pool=False)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SubsequentStage(nn.Module):
    def __init__(self, n_maps: int, img_feat_ch: int, n_base_ch: int):
        super().__init__()

        self.conv_1 = Conv2dBlock(n_base_ch, img_feat_ch, ker_size=5, pad=2, act=True, pool=False)

        self.blocks = nn.ModuleList([
            Conv2dBlock(n_maps + img_feat_ch, n_base_ch, ker_size=11, pad=5, act=True, pool=False),
            Conv2dBlock(n_base_ch, n_base_ch, ker_size=11, pad=5, act=True, pool=False),
            Conv2dBlock(n_base_ch, n_base_ch, ker_size=11, pad=5, act=True, pool=False),
            Conv2dBlock(n_base_ch, n_base_ch, ker_size=1, pad=0, act=True, pool=False),
            Conv2dBlock(n_base_ch, n_maps, ker_size=1, pad=0, act=False, pool=False)
        ])

    def forward(self, x: Tensor, img_ref: Tensor) -> Tensor:
        img_ref = self.conv_1(img_ref)
        x = torch.cat((x, img_ref), 1)
        for block in self.blocks:
            x = block(x)
        return x


class ConvolutionalPoseMachines(nn.Module):

    def __init__(self,
                 n_keypoints: int,
                 n_substages: int,
                 n_base_ch: int = 64,
                 img_feat_ch: int = 16,
                 include_bground_map: bool = False,
                 norm_layer: nn = nn.LayerNorm,
                 act_layer: nn = nn.GELU):
        super().__init__()

        self.init_stage = InitialStage(
            n_keypoints + include_bground_map,
            n_base_ch
        )
        self.img_feat = ImageFeatureBlock(n_base_ch)

        self.subsequent_stages_list = nn.ModuleList(
            [SubsequentStage(
                n_keypoints + include_bground_map,
                img_feat_ch=img_feat_ch,
                n_base_ch=n_base_ch
            ) for i in range(n_substages)]
        )

    def forward(self, x: Tensor) -> Tensor:
        img_ref = self.img_feat(x)
        outputs = [self.init_stage(x)]

        for sub_stage in self.subsequent_stages_list:
            outputs.append(sub_stage(outputs[-1], img_ref))

        return torch.stack(outputs, dim=1)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input = torch.rand((5, 3, 368, 368), device=device)
    print('Input shape:', input.shape)

    with torch.inference_mode():
        model = ConvolutionalPoseMachines(
            n_keypoints=16,
            n_substages=3,
            n_base_ch=128,
            img_feat_ch=32
        ).to(device)
        output = model(input)
        print('Output shape:', output.shape)
    for c in model.children():
        print(c)
