import torch.nn as nn


class SimplePose(nn.Module):

    def __init__(self, keypoints_number=16, include_visibility=True):
        super().__init__()

        self.keypoints_number = keypoints_number
        self.include_visibility = include_visibility

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=256 * 16 * 16, out_features=1024)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=1024, out_features=1016)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=1016, out_features=self.keypoints_number * (2 + self.include_visibility))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)

        return x
