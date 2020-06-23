import torch
import torch.nn as nn
import torch.nn.functional as functional
from base import BaseModel

import utils
from utils import rel_to_abs, get_descriptor_by_pos, get_descriptor_by_pos_batch, brute_force_match

def vgg_block(in_channels, out_channels, pooling=True):
    max_pool = []
    if pooling:
        max_pool = [nn.MaxPool2d(kernel_size=2)]

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=False),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=False),
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
    """ Pytorch definition of KeyPointNet Network.
        the structure from Jiexiong Tang (https://arxiv.org/abs/1912.10615)
     """

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

    def forward(self, image):
        """ Forward pass that jointly computes score, location and descriptor
        tensors.

        Input
        image: Image pytorch tensor shaped B x 3 x H x W.
        Output
        score: Output score pytorch tensor shaped B x 1 × H/8 × W/8
        location: Output point pytorch tensor shaped B x 2 x H/8 x W/8.
        descriptor: Output descriptor pytorch tensor shaped B x 256 x H/8 x W/8.
        """
        B, _ , H, W = image.shape

        # Shared Encoder.
        f8 = self.encoder_frontend(image)
        f11 = self.encoder_backend(f8)
        
        # Score Head.
        S = self.score_head(f11)    # B x 1 x H/8 x W/8

        # Location Head.
        Prel = self.location_head(f11) # B x 2 x H/8 x W/8

        # Descriptor Head.
        f18 = self.descriptor_head_frontend(f11)
        F = self.descriptor_head_backend(torch.cat((f8, f18), dim=1))   # B x 256 x H/4 x W/4

        # Convert local (relative) positions P to global pixel positions
        P = rel_to_abs(Prel)         # B x 2 x H/8 x W/8
               
        # flatten
        Sflat = S.view(B, -1)           # B x N (N = H/8 * W/8)
        Pflat = P.view(B, 2, -1)        # B x 2 x N
        Prelflat = Prel.view(B, 2, -1)  # B x 2 x N
        Fflat = F.view(B, 256, -1)      # B x 256 x N(or 4N)

        # choice 1: downsample
        # F = functional.interpolate(F, (H//8, W//8), mode='bicubic', align_corners=True)   # B x 256 x H/8 x W/8
        # Fflat = F.view(B, 256, -1)      # B x 256 x N(or 4N)

        # choice 2: upsample
        # F = functional.interpolate(F, (H, W), mode='bicubic', align_corners=True)   # B x 256 x H x W
        # Pl = P.long()
        # F = torch.stack([F[i, :, Pl[i, 1], Pl[i, 0]] for i in range(B)], dim=0)
        # Fflat = F.view(B, 256, -1)      # B x 256 x N(or 4N)

        output = {
            "S": Sflat,
            "P": Pflat,
            "F": Fflat,
            "Prel": Prelflat,
        }

        return output

class IONet(BaseModel):
    """ Pytorch definition of IONet Network. 
        the structure from Jiexiong Tang (https://arxiv.org/abs/1912.10615)
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
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
        # f1 = self.relu(self.conv1a(x))
        f1 = functional.relu(self.conv1a(x))
        # 2
        f2 = self.rb2(f1)
        # 3
        f3 = self.rb3(torch.add(f2, f1))
        # 4
        f4 = self.rb4(torch.add(f3, f2))
        # 5
        f5 = self.rb5(torch.add(f4, f3))
        # 6
        R = self.conv6a(f5)
        return R

class KP2DNet(BaseModel):
    """ Pytorch definition of entire Network. 
        the structure from Jiexiong Tang (https://arxiv.org/abs/1912.10615)
    """

    def __init__(self):
        super().__init__()
        self.image_shape = (3, 240, 320)

        self.KPN = KeyPointNet()
        self.ION = IONet()

    def forward(self, img, warp, homog):
        """Forward pass through the KeyPointNet and IONet.

        Args:
            img (pytorch tensor shaped B x 3 x H x W): 3-channel images
            warp (pytorch tensor shaped B x 3 x H x W): 3-channel images warped by homography matrix
            homog (pytorch tensor shaped B x ? x ?) : homography matrixs transform 'img' to 'warp'

        Returns:
            ouputs (data dictionary) : data used to calculate the loss function
        """        
        A = self.KPN(img)
        B = self.KPN(warp)
        matches = self.match_by_descriptor(A, B)
        # pytorch tensor shaped B x 1 x N : the probability that a point-pair belongs to an inlier set
        R = self.ION(matches)

        outputs = {
            'A': A,
            'B': B,
            'PP': matches,
            'R': R,
            'H': homog,
        }
        return outputs

    # def match_by_descriptor(self, A, B):
    #     AS = A['S']         # B x N (N = H/8 * W/8)
    #     AP = A['P']         # B x 2 x N
    #     AF = A['F']         # B x 256 x 4N, 'fs' in paper

    #     BS = B['S']         # B x N (N = H/8 * W/8)
    #     BP = B['P']         # B x 2 x N
    #     BF = B['F']         # B x 256 x 4N, 'ft' in paper

    #      # Get data with lowest K score (S)
    #     k = 300
    #     _, ids = torch.topk(AS, k, dim=1, largest=False, sorted=False)                  # B x K
    #     APmax = torch.stack([AP[i, :, ids[i]] for i in range(ids.shape[0])], dim=0)     # B x 2 x K
        
    #     AFmax, _ = get_descriptor_by_pos_batch(APmax, AF, self.image_shape)             # B x 256 x k

    #     _, ids = torch.topk(BS, k, dim=1, largest=False, sorted=False)                  # B x 2 x K
    #     BPmax = torch.stack([BP[i, :, ids[i]] for i in range(ids.shape[0])], dim=0)     # B x 2 x K
        
    #     BFmax, _ = get_descriptor_by_pos_batch(BPmax, BF, self.image_shape)             # B x 256 x k

    #     dists, Bids = brute_force_match(AFmax, BFmax)                                   # both B x K
    #     Bmatch = torch.stack([BP[i, :, Bids[i]] for i in range(Bids.shape[0])], dim=0)  # B x 2 x K

    #     matches = torch.cat([APmax, Bmatch, dists.unsqueeze(1)], dim=1)                 # B x 5 x K

    #     return matches  # B x 5 x K
    
    def match_by_descriptor(self, A, B):
        AS = A['S']         # B x N (N = H/8 * W/8)
        AP = A['P']         # B x 2 x N
        AF = A['F']         # B x 256 x N, 'fs' in paper

        BS = B['S']         # B x N (N = H/8 * W/8)
        BP = B['P']         # B x 2 x N
        BF = B['F']         # B x 256 x N, 'ft' in paper

         # Get data with lowest K score (S)
        k = 300
        _, ids = torch.topk(AS, k, dim=1, largest=False, sorted=False)                  # B x K
        APmax = torch.stack([AP[i, :, ids[i]] for i in range(ids.shape[0])], dim=0)     # B x 2 x K
        AFmax = torch.stack([AF[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)        # B x 256 x K

        _, ids = torch.topk(BS, k, dim=1, largest=False, sorted=False)                  # B x 2 x K
        # BPmax = torch.stack([BP[i, :, ids[i]] for i in range(ids.shape[0])], dim=0)     # B x 2 x K
        BFmax = torch.stack([BF[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)       # B x 256 x K

        dists, Bids = brute_force_match(AFmax, BFmax)                                   # both B x K
        BPmatch = torch.stack([BP[i, :, Bids[i]] for i in range(Bids.shape[0])], dim=0)  # B x 2 x K

        matches = torch.cat([APmax, BPmatch, dists.unsqueeze(1)], dim=1)                 # B x 5 x K

        return matches  # B x 5 x K


# from https://github.com/ErikOrjehag/sfmnet/blob/5b4001b3950937f604bd394ec2bb14199c1c56d7/networks/unsuperpoint.py
def conv(in_channels, out_channels):
    # Create conv2d layer
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

def bb_conv(in_channels, out_channels, last_layer=False):
    # Create 2 conv layers separated by batch norm and leaky relu
    if last_layer:
        last = []
    else:
        last = [nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(inplace=True)]
    return nn.Sequential(
        conv(in_channels, out_channels),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(inplace=True),
        conv(out_channels, out_channels),
        *last
    )

class UnsuperPoint(BaseModel):

    def __init__(self, N):
        super().__init__()
        self.N = N
        self.backbone = nn.Sequential(
            bb_conv(3, 32),
            nn.MaxPool2d(kernel_size=2),
            bb_conv(32, 64),
            nn.MaxPool2d(kernel_size=2),
            bb_conv(64, 128),
            nn.MaxPool2d(kernel_size=2),
            bb_conv(128, 256, last_layer=True),
        )

        self.score_decoder = nn.Sequential(
            conv(256, 256),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True),
            conv(256, 1),
            nn.Sigmoid(),
        )

        self.position_decoder = nn.Sequential(
            conv(256, 256),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True),
            conv(256, 2),
            nn.Sigmoid(),
        )

        self.descriptor_decoder = nn.Sequential(
            conv(256, 256),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True),
            conv(256, 256),
        )

    def forward(self, image):

        B, _, H, W = image.shape
        image = utils.normalize_image(image)
        
        # CNN (joint backbone, separate decoder heads)
        features = self.backbone(image)
        S = self.score_decoder(features)
        Prel = self.position_decoder(features)
        F = self.descriptor_decoder(features)
        
        # Relative to absolute pixel coordinates
        P = self.rel_to_abs(Prel)
        
        # Flatten
        Sflat = S.view(B, -1)
        Pflat = P.view(B, 2, -1)
        Prelflat = Prel.view(B, 2, -1)
        Fflat = F.view(B, 256, -1)

        # Get data with top N score (S)
        Smax, ids = torch.topk(Sflat, k=self.N, dim=1, largest=True, sorted=False)
        Pmax = torch.stack([Pflat[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)
        Fmax = torch.stack([Fflat[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)

        outputs = {
            "S": Smax,
            "P": Pmax,
            "Prel": Prelflat,
            "F": Fmax,
        }

        return outputs

    def rel_to_abs(self, P):
        # Convert local (relative) positions P to global pixel positions
        B, _, H, W = P.shape
        cols = torch.arange(0, W, device=P.device).view(1, 1, W).expand(B, H, W)
        rows = torch.arange(0, H, device=P.device).view(1, H, 1).expand(B, H, W)
        return (P + torch.stack((cols, rows), dim=1)) * 8

class SiameseUnsuperPoint(BaseModel):

    def __init__(self, N=200):
        super().__init__()
        self.unsuperpoint = UnsuperPoint(N=N)

    def forward(self, img, warp, homog):
        A = self.unsuperpoint(img)
        B = self.unsuperpoint(warp)
        outputs = {
            "A": A,
            "B": B,
            'img':img,
            'warp': warp,
            "homography":homog,
        }
        return outputs
