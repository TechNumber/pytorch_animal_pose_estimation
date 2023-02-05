import torch


class InitialStage(torch.nn.Module):
    def __init__(self, number_of_maps):
        super().__init__()
        self.number_of_maps = number_of_maps

        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, padding=4)
        self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, padding=4)
        self.conv_4 = torch.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, padding=2)
        self.conv_5 = torch.nn.Conv2d(in_channels=16, out_channels=256, kernel_size=9, padding=4)
        self.conv_6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv_7 = torch.nn.Conv2d(in_channels=256, out_channels=self.number_of_maps, kernel_size=1)

        self.act = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv_2(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv_3(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv_4(x)
        x = self.act(x)

        x = self.conv_5(x)
        x = self.act(x)

        x = self.conv_6(x)
        x = self.act(x)

        x = self.conv_7(x)

        return x


class ImageFeature(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, padding=4)
        self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, padding=4)

        self.act = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv_2(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv_3(x)
        x = self.act(x)
        x = self.pool(x)

        return x


class SubsequentStage(torch.nn.Module):
    def __init__(self, number_of_maps, image_feat_channels):
        super().__init__()
        self.number_of_maps = number_of_maps
        self.image_feat_channels = image_feat_channels

        self.conv_1 = torch.nn.Conv2d(in_channels=64, out_channels=self.image_feat_channels, kernel_size=5, padding=2)
        self.conv_2 = torch.nn.Conv2d(in_channels=self.number_of_maps + self.image_feat_channels, out_channels=64, kernel_size=11, padding=5)
        self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=11, padding=5)
        self.conv_4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=11, padding=5)
        self.conv_5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv_6 = torch.nn.Conv2d(in_channels=64, out_channels=self.number_of_maps, kernel_size=1)

        self.act = torch.nn.ReLU()

    def forward(self, x, x_reference):
        x_reference = self.conv_1(x_reference)
        x_reference = self.act(x_reference)

        x = torch.cat((x, x_reference), 1)

        x = self.conv_2(x)
        x = self.act(x)

        x = self.conv_3(x)
        x = self.act(x)

        x = self.conv_4(x)
        x = self.act(x)

        x = self.conv_5(x)
        x = self.act(x)

        x = self.conv_6(x)

        return x


class ConvolutionalPoseMachines(torch.nn.Module):

    def __init__(self,
                 keypoints,
                 sub_stages,
                 include_bground_map=False,
                 device=torch.device('cpu')):
        super().__init__()
        self.sub_stages = sub_stages
        self.keypoints = keypoints
        self.include_bground_map = include_bground_map

        self.init_stage = InitialStage(self.keypoints + self.include_bground_map).to(device)
        self.image_feat = ImageFeature().to(device)

        self.subsequent_stages_list = torch.nn.ModuleList()
        for i in range(sub_stages):
            self.subsequent_stages_list.append(
                SubsequentStage(
                    number_of_maps=self.keypoints + self.include_bground_map,
                    image_feat_channels=16).to(device)
            )

    def forward(self, x):

        # TODO: cat vs on-the-fly loss calculation

        x_reference = self.image_feat(x)
        outputs = [self.init_stage(x)]

        for sub_stage in self.subsequent_stages_list:
            outputs.append(sub_stage(outputs[-1], x_reference))

        return torch.stack(outputs, dim=1)
