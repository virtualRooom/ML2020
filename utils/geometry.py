import torch
import torch.nn.functional as F

def rel_to_abs(Prel, sigma1=2.0, sigma2=8.0):
    """Convert local (relative) positions P to global pixel positions
       B: batch size, H: image height, W: image width

    Arguments:
        Prel {pytorch tensor shaped B * 2 * H/8 * W/8} -- key point relative coordinate tensor

    Keyword Arguments:
        sigma1 {float} -- ratio relative to the cell size (default: {2.0})
        sigma2 {float} -- the cell size (default: {8.0})

    Returns:
        {pytorch tensor shaped B * 2 * H/8 * W/8} -- key point absolute coordinate tensor
    """    

    # scale constant
    c = sigma1 * (sigma2 - 1) / 2

    B, _, H, W = Prel.shape
    # center of each cell
    cols = torch.arange((sigma2 - 1) / 2, sigma2 * W, sigma2,
                        device=Prel.device).view(1, 1, W).expand(B, H, W)

    rows = torch.arange((sigma2 - 1) / 2, sigma2 * H, sigma2,
                        device=Prel.device).view(1, H, 1).expand(B, H, W)
    
    cols = c * Prel[:, 0, :, :] + cols
    rows = c * Prel[:, 1, :, :] + rows

    cols = torch.clamp(cols, min=0.01, max=W*8.-.01)
    rows = torch.clamp(rows, min=0.01, max=H*8.-.01)

    P = torch.stack((cols, rows), dim=1)
    return P

def pose_vec2mat(vec):
    t = vec[:,:3].unsqueeze(-1) # [B, 3, 1]
    r = vec[:,3:]
    R = euler2mat(r) # [B, 3, 3]
    transform = torch.cat([R, t], dim=2) # [B, 3, 4]
    return transform

def to_homog_coords(coords):
    pad = (0, 0, 0, 0, 0, 1) # [B,C,H,W]
    if len(coords.shape) == 3: # [B,C,N]
        pad = pad[2:]
    return F.pad(input=coords, pad=pad, mode="constant", value=1)

def to_homog_matrix(matrix):
    matrix = F.pad(input=matrix, pad=(0, 0, 0, 1), mode="constant", value=0)
    matrix[...,-1,-1] = 1.0
    return matrix

def from_homog_coords(coords):
    X = coords[:,0]
    Y = coords[:,1]
    Z = coords[:,2]#.clamp(min=1e-3)
    return torch.stack((X/Z, Y/Z), dim=1)

def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat
