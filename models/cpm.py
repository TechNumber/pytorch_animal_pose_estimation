import torch.nn as nn
import torch.nn.functional as F
import torch


class CPM(nn.Module):
    def __init__(self):
        super(CPM, self).__init__()
        self.stage1 = CPMStage1()
        self.stage2_image = CPMStage2Image()
        self.stage2 = CPMStageT()
        self.stage3 = CPMStageT()
        self.stage4 = CPMStageT()
        self.stage5 = CPMStageT()
        self.stage6 = CPMStageT()

    def forward(self, image, center_map):
        stage1_maps = self.stage1(image)
        stage2image_maps = self.stage2_image(image)
        stage2_maps = self.stage2(stage1_maps, stage2image_maps, center_map)
        stage3_maps = self.stage3(stage2_maps, stage2image_maps, center_map)
        stage4_maps = self.stage4(stage3_maps, stage2image_maps, center_map)
        stage5_maps = self.stage5(stage4_maps, stage2image_maps, center_map)
        stage6_maps = self.stage6(stage5_maps, stage2image_maps, center_map)

        return stage1_maps, stage2_maps, stage3_maps, stage4_maps, stage5_maps, stage6_maps


class CPMStage1(nn.Module):
    def __init__(self):
        super(CPMStage1, self).__init__()
        self.k = 14

        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7 = nn.Conv2d(512, self.k + 1, kernel_size=1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)

        return x


class CPMStage2Image(nn.Module):
    def __init__(self):
        super(CPMStage2Image, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        return x


class CPMStageT(nn.Module):
    def __init__(self):
        super(CPMStageT, self).__init__()
        self.k = 14

        self.conv_image = nn.Conv2d(self.k + 1, self.k + 1, kernel_size=5, padding=2)

        self.conv1 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

    def forward(self, stage1_maps, stage2image_maps, center_map):

        x = F.relu(self.conv_image(stage1_maps))
        x = torch.cat([stage2image_maps, x, center_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        return x
