import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class KeyPointNet(BaseModel):
    """ Pytorch definition of KeyPointNet Network. """

    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.pixelShuffle = nn.PixelShuffle(2)
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.conv4a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(256)
        # Score Head. TODO: determine final conv2d kernel_size and padding
        self.convSa = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bnSa = nn.BatchNorm2d(256)
        self.convSb = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        # Location Head. TODO: determine final conv2d kernel_size and padding
        self.convLa = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bnLa = nn.BatchNorm2d(256)
        self.convLb = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        # Descriptor Head. TODO: determine final conv2d kernel_size and padding
        self.convD1a = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bnD1a = nn.BatchNorm2d(256)
        self.convD1b = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bnD1b = nn.BatchNorm2d(512)

        self.convD2a = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bnD2a = nn.BatchNorm2d(256)
        self.convD2b = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
        x: Image pytorch tensor shaped N x 3 x H x W.
        Output
        score: Output score pytorch tensor shaped N x 1 × H/8 × W/8
        location: Output point pytorch tensor shaped N x 2 x H/8 x W/8.
        descriptor: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        # 1
        x = self.relu(self.bn1a(self.conv1a(x)))
        # 2
        feature2 = self.dropout(self.relu(self.bn1b(self.conv1b(x))))
        # 3
        x, ind1 = self.pool(feature2)
        # 4 
        x = self.relu(self.bn2a(self.conv2a(x)))
        # 5
        feature5 = self.dropout(self.relu(self.bn2b(self.conv2b(x))))
        # 6
        x , ind2 = self.pool(feature5)
        # 7
        x = self.relu(self.bn3a(self.conv3a(x)))
        # 8
        feature8 = self.dropout(self.relu(self.bn3b(self.conv3b(x))))
        # 9
        x, ind3 = self.pool(feature8)
        # 10
        x = self.relu(self.bn4a(self.conv4a(x)))
        # 11
        feature11 = self.dropout(self.relu(self.bn4b(self.conv4b(x))))
        
        # Score Head.
        # 12
        sx = self.dropout(self.bnSa(self.convSa(feature11)))
        # 13
        score = self.sigmoid(self.convSb(sx))
        # Location Head.
        # 14 
        lx = self.dropout(self.bnLa(self.convLa(feature11)))
        # 15
        location = self.tanh(self.convLb(lx))
        # Descriptor Head.
        # 16
        dx = self.dropout(self.bnD1a(self.convD1a(feature11)))
        # 17
        dx = self.bnD1b(self.convD1b(dx))
        # 18
        feature18 = self.pixelShuffle(dx)
        19
        dx = self.bnD2a(self.convD2a(torch.cat((feature8, feature18), dim=1)))
        # 20
        descriptor = self.convD2b(dx)

        output = {'score': score, 'location': location, 'descriptor': descriptor}
        return output