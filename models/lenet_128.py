import torch


class LeNet128(torch.nn.Module):
    def __init__(self,
                 conv_size=3,
                 activation='relu',
                 pooling='max',
                 use_batch_norm=False,
                 keep_size=True):
        super().__init__()
        self.conv_size = conv_size
        self.activation = activation
        self.pooling = pooling
        self.use_batch_norm = use_batch_norm
        self.keep_size = keep_size

        if conv_size == 5:
            if keep_size:
                self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
                self.conv2_1 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2)
                self.conv3_1 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
            else:
                self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)
                self.conv2_1 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
                self.conv3_1 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0)
        elif conv_size == 3:
            if keep_size:
                self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
                self.conv1_2 = torch.nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)
                self.conv2_1 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
                self.conv2_2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
                self.conv3_1 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
                self.conv3_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
            else:
                self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=0)
                self.conv1_2 = torch.nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=0)
                self.conv2_1 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=0)
                self.conv2_2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0)
                self.conv3_1 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
                self.conv3_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        else:
            raise NotImplemented

        if keep_size:
            self.fc1 = torch.nn.Linear(in_features=8192, out_features=256)
        else:
            self.fc1 = torch.nn.Linear(in_features=4608, out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=128)
        self.fc3 = torch.nn.Linear(in_features=128, out_features=48)

        if use_batch_norm:
            self.norm0 = torch.nn.BatchNorm2d(num_features=3)
            self.norm1 = torch.nn.BatchNorm2d(num_features=self.conv1_1.out_channels)
            self.norm2 = torch.nn.BatchNorm2d(num_features=self.conv2_1.out_channels)
            self.norm5 = torch.nn.BatchNorm1d(num_features=self.conv3_1.out_features)
            self.norm3 = torch.nn.BatchNorm1d(num_features=self.fc1.out_features)
            self.norm4 = torch.nn.BatchNorm1d(num_features=self.fc2.out_features)

        if activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif activation == 'relu':
            self.act = torch.nn.ReLU()
        else:
            raise NotImplemented

        if pooling == 'avg':
            self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == 'max':
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplemented

    def forward(self, x):
        if self.use_batch_norm:
            x = self.norm0(x)

        x = self.conv1_1(x)
        if self.conv_size == 3:
            x = self.conv1_2(x)
        x = self.act(x)
        if self.use_batch_norm:
            x = self.norm1(x)
        x = self.pool(x)

        x = self.conv2_1(x)
        if self.conv_size == 3:
            x = self.conv2_2(x)
        x = self.act(x)  # Поменять местами нормализацию и функцию активации
        if self.use_batch_norm:
            x = self.norm2(x)
        x = self.pool(x)

        x = self.conv3_1(x)
        if self.conv_size == 3:
            x = self.conv3_2(x)
        x = self.act(x)  # Поменять местами нормализацию и функцию активации
        if self.use_batch_norm:
            x = self.norm3(x)
        x = self.pool(x)

        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc1(x)
        x = self.act(x)
        if self.use_batch_norm:
            x = self.norm4(x)

        x = self.fc2(x)
        x = self.act(x)
        if self.use_batch_norm:
            x = self.norm5(x)

        x = self.fc3(x)

        return x
