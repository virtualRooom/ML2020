import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils import rel_to_abs

def vgg_block(in_channels, out_channels, pooling=True):
    max_pool = []
    if pooling:
        max_pool = [nn.MaxPool2d(kernel_size=2)]

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Dropout2d(p=0.2, inplace=True),
        *max_pool
    )

def residual_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
        nn.InstanceNorm1d(out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
        nn.InstanceNorm1d(out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )

class KeyPointNet(BaseModel):
    """ Pytorch definition of KeyPointNet Network. """

    def __init__(self):
        super().__init__()

        # Shared Encoder.
        self.encoder_frontend = nn.Sequential(
            vgg_block(3,32),
            vgg_block(32,64),
            vgg_block(64,128, pooling=False),
        )
        self.encoder_backend = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            vgg_block(128, 256, pooling=False)
        )
       
        # Score Head.
        self.score_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        
        # Location Head.
        self.location_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=0.2, inplace=True),
            nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        # Descriptor Head.
        self.descriptor_head_frontend = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PixelShuffle(2),
        )
        self.descriptor_head_backend = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )

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
        f8 = self.encoder_frontend(x)
        f11 = self.encoder_backend(f8)
        
        # Score Head.
        S = self.score_head(f11)    # B x 1 x H/8 x W/8

        # Location Head.
        P = self.location_head(f11) # B x 2 x H/8 x W/8

        # Descriptor Head.
        f18 = self.descriptor_head_frontend(f11)
        F = self.descriptor_head_backend(torch.cat((f8, f18), dim=1))   # B x 256 x H/4 x W/4

        # Convert local (relative) positions P to global pixel positions
        Prel = rel_to_abs(P)         # B x 2 x H/8 x W/8

        # flatten
        B, _, H, W = x.shape
        Sflat = S.view(B, -1)           # B x N (N = H/8 * W/8)
        Pflat = P.view(B, 2, -1)        # B x 2 x N
        Fflat = F.view(B, 256, -1)      # B x 256 x 4N
        Prelflat = Prel.view(B, 2, -1)  # B x 2 x N

        # # Get data with top K score (S)
        # Smax, ids = torch.topk(Sflat, k=self.N, dim=1, largest=True, sorted=False)              # B x K(N = top K = 300)
        # Pmax = torch.stack([Pflat[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)             # B x 2 x K
        # #Prelmax = torch.stack([Prelflat[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)
        # Fmax = torch.stack([Fflat[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)             # B x 256 x K 

        output = {
            "S": Sflat,
            "P": Pflat,
            "F": Fflat,
            "Prel": Prelflat,
        }

        return output

class IONet(BaseModel):
    """ Pytorch definition of IONet Network. 
        the structure from Brachmann & Rother (https://arxiv.org/abs/1905.04132)
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # 1
        self.conv1a = nn.Conv1d(5, 128, kernel_size=1, stride=1)
        # 2
        self.rb2 = residual_block(128, 128)
        # 3
        self.rb3 = residual_block(128, 128)
        # 4
        self.rb4 = residual_block(128, 128)
        # 5
        self.rb5 = residual_block(128, 128)
        # 6
        self.conv6a = nn.Conv1d(128, 1, kernel_size=1, stride=1)


    def forward(self, x):
        """ Forward pass that .
        Input
        x: 5-dimensional (consists of keypoint pair and descriptor distance)
        vector pytorch tensor shaped 5 x N.
        Output
        label: binary inlier-outlier classification pytorch tensor shaped 1 x N
        """
        # 1
        f1 = self.relu(self.conv1a(x))
        # 2
        f2 = self.rb2(f1)
        # 3
        f3 = self.rb3(torch.add(f2, f1))
        # 4
        f4 = self.rb4(torch.add(f3, f2))
        # 5
        f5 = self.rb5(torch.add(f4, f3))
        # 6
        label = self.conv6a(f5)
        return label
