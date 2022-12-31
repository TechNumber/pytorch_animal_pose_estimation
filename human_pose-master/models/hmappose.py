import torch.nn as nn
import torch.nn.functional as F

class HmapPose(nn.Module):
    
    def __init__(self):
        super(HmapPose, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, padding = 2)
        self.conv2 = nn.Conv2d(128, 128, 5, padding = 2)
        self.conv3 = nn.Conv2d(128, 128, 5, padding = 2)
        self.conv33 = nn.Conv2d(128, 512, 9, padding = 4)
        self.conv4 = nn.Conv2d(512, 256, 1)
        self.conv5 = nn.Conv2d(256, 256, 1)
        self.conv6 = nn.Conv2d(256, 15, 1)
    
    def forward(self, x):
        # x -> [-1, 3, 256, 256]
        x = F.relu(self.conv1(x))
        # x -> [-1, 32, 256, 256]
        x = F.max_pool2d(x, 2)
        # x -> [-1, 32, 128, 128]
        
        x = F.relu(self.conv2(x))
        # x -> [-1, 32, 128, 128]
        x = F.max_pool2d(x, 2)
        # x -> [-1, 32, 64, 64]
        
        x = F.relu(self.conv3(x))
        # x -> [-1, 64, 64, 64]
        x = F.max_pool2d(x, 2)
        # x -> [-1, 64, 32, 32]
        x = F.relu(self.conv33(x))
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.tanh(self.conv6(x))
        # x -> [-1, 14, 32, 32]
        return x