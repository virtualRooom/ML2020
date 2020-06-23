import torch
import torch.nn.functional as F

import base

from utils import from_homog_coords, to_homog_coords, get_descriptor_by_pos, discard_outside_points

def kpn_loss(A, B, homog): 
    AS = A['S']         # B x N (N = H/8 * W/8)
    AP = A['P']         # B x 2 x N
    AF = A['F']         # B x 256 x 4N, 'fs' in paper
    APrel = A['Prel']

    BS = B['S']         # B x N (N = H/8 * W/8)
    BP = B['P']         # B x 2 x N
    BF = B['F']         # B x 256 x 4N, 'ft' in paper
    BPrel = B['Prel']

    # TODO - get True image shape
    img_size = (3, 240, 320)

    # 1. Calculate location loss
    B = AP.shape[0]
    _, image_h, image_w = img_size
    
    T = torch.tensor([
            [1, 0, image_w/2.],
            [0, 1, image_h/2.],
            [0, 0, 1],
        ], dtype=torch.float, requires_grad=True).repeat(B,1,1).to(homog.device)
    # Points from branch A transformed by homography        
    APh = from_homog_coords(T @ homog @ torch.inverse(T) @ to_homog_coords(AP)) # B x 2 x N tensor

    # Matrix of distances between points
    D = (APh.permute(0,2,1).unsqueeze(2) - BP.permute(0,2,1).unsqueeze(1)).norm(dim=-1) # B x N x N tensor

    # Matching, create ids which maps the B points to its closest A point (A[ids] <-> B)
    Dmin, ids = torch.min(D, dim=1) # both Dmin and ids shaped B x N

    # Create a mask for only the maped ids that are closer than a threshold.
    threshold = 1.0  # epilon_uv
    mask = Dmin.le(threshold)   #  B x N tensor

    d = Dmin[mask]  # B x M(M<N) vector
    dmean = d.mean()

    loc_loss = d.sum()

    # 2. Calculate score loss
    mask_ = mask.view(-1)   # B x N vector
    ids_ = ids.view(-1)     # B x N vector
    AS_ = AS.view(-1)[ids_][mask_] # see debugger_point
    BS_ = BS.view(-1)[mask_]

    # Scores of corresonding points should be similar to each other
    l_score = (AS_ - BS_) ** 2

    # Increase score if they are near, supress score if they are far
    S_ = (AS_ + BS_) / 2
    l_usp = S_ * (d - dmean)
    
    score_loss = l_score.sum() + l_usp.sum()

    # 3. Calculate descriptor loss

    desc_loss = 0.
    for b in range(B):
        APh_b = APh[b] # 2 x N tensor
        AF_b = AF[b]
        BF_b = BF[b]

        # get warped points inside the image boundary, reduce N to M
        APh_b, mask = discard_outside_points(APh_b, img_size) # reduced to 2 x M tensor

        AP_b = AP[b].masked_select(mask).reshape(2,-1)

        # get descriptor of source key points
        fs, _ = get_descriptor_by_pos(AP_b, AF_b, img_size)       # 256 x M tensor, key point's descriptor in source image
        fpos, fpos_idx = get_descriptor_by_pos(APh_b, BF_b, img_size)     # 256 x M tensor, 1xM vector
        fpos_norm = (fs-fpos).norm(dim=0)

        # Find the cloest descriptor of 'fs' in target descriptor map(BF)
        FD = (fs.permute(1,0).unsqueeze(1) - BF_b.permute(1,0).unsqueeze(0)).norm(dim=-1) # M x 4N tensor

        # get index of top two cloest descriptor(potential negative sample)
        _, fneg_idxs = torch.topk(FD, 2, largest=False, sorted=False)  # both M x 2 tensor
        mask = fneg_idxs[:,0]==fpos_idx # 1xM vector
        mask = torch.stack([mask,~mask],dim=-1) # M x 2 tensor
        # get index of negative sample
        fneg_idx = fneg_idxs[mask]  # 1xM vector

        # get negative descriptor by index
        fneg = BF_b[:,fneg_idx]               # 256 x M
        fneg_norm = (fs-fneg).norm(dim=0)   # M tensor

        # distance margin parameter
        m = 0.2
        l_dsp = torch.max(torch.zeros_like(fpos_norm), fpos_norm-fneg_norm + m)

        desc_loss += l_dsp.sum()

    del FD

    # 4. final loss
    alpha, beta, lamda = 1.0, 2.0, 1.0 # weights balancing different losses(parameters)
    loss = alpha * loc_loss + beta * desc_loss + lamda * score_loss
    
    return loss

def kpn_loss2(A, B, homog): 
    AS = A['S']         # B x N (N = H/8 * W/8)
    AP = A['P']         # B x 2 x N
    AF = A['F']         # B x 256 x 4N, 'fs' in paper

    BS = B['S']         # B x N (N = H/8 * W/8)
    BP = B['P']         # B x 2 x N
    BF = B['F']         # B x 256 x 4N, 'ft' in paper

    # TODO - get True image shape
    img_size = (3, 240, 320)

    # 1. Calculate location loss
    B = AP.shape[0]
    _, image_h, image_w = img_size
    
    T = torch.tensor([
            [1, 0, image_w/2.],
            [0, 1, image_h/2.],
            [0, 0, 1],
        ], dtype=torch.float, requires_grad=True).repeat(B,1,1).to(homog.device)
    # Points from branch A transformed by homography        
    APh = from_homog_coords(T @ homog @ torch.inverse(T) @ to_homog_coords(AP)) # B x 2 x N tensor

    # Matrix of distances between points
    D = (APh.permute(0,2,1).unsqueeze(2) - BP.permute(0,2,1).unsqueeze(1)).norm(dim=-1) # B x N x N tensor

    # Matching, create ids which maps the B points to its closest A point (A[ids] <-> B)
    Dmin, ids = torch.min(D, dim=1) # both Dmin and ids shaped B x N

    # Create a mask for only the maped ids that are closer than a threshold.
    threshold = 1.0  # epilon_uv
    mask = Dmin.le(threshold)   #  B x N tensor

    d = Dmin[mask]  # B x M(M<N) vector
    dmean = d.mean()

    loc_loss = d.sum()

    # 2. Calculate score loss
    mask_ = mask.view(-1)   # B x N vector
    ids_ = ids.view(-1)     # B x N vector
    AS_ = AS.view(-1)[ids_][mask_] # see debugger_point
    BS_ = BS.view(-1)[mask_]

    # Scores of corresonding points should be similar to each other
    l_score = (AS_ - BS_) ** 2

    # Increase score if they are near, supress score if they are far
    S_ = (AS_ + BS_) / 2
    l_usp = S_ * (d - dmean)
    
    score_loss = l_score.sum() + l_usp.sum()

    # 3. Calculate descriptor loss
    desc_loss = 0

    # 4. final loss
    alpha, beta, lamda = 1.0, 2.0, 1.0 # weights balancing different losses(parameters)
    loss = alpha * loc_loss + beta * desc_loss + lamda * score_loss
    
    return loss


def ion_loss(PP, R):
    """loss function of IONet

    Arguments:
        PP {pytorch tensor shaped B x 5 x N} -- point pairs(only keypoints with the lowest K predicted scores are used for training.)
        R {pytorch tensor shaped B x 1 x N} -- the probability that each pair is an inlier or outlier(the output of the IO-Net)

    Returns:
        float -- the loss of IONet
    """
    # TODO - move global parameter to config.json
    epsilon_uv = 1.0

    # get keypoints from point pairs
    Ps, Pt_star = PP[:, 0:2, :], PP[:, 2:4, :]  # Bx2xN tensor

    tmp = torch.sign((Ps-Pt_star).norm(dim=1).unsqueeze(1)-epsilon_uv) #Bx1xN tensor

    l_io = (R-tmp).norm(dim=1)
    loss_io = 0.5 * l_io.sum()

    return loss_io

def KP2D_loss(data):
    loss = kpn_loss(data['A'], data['B'], data['H']) + ion_loss(data['PP'], data['R'])
    return loss

def V3_loss(data):
    loss = kpn_loss2(data['A'], data['B'], data['H']) + ion_loss(data['PP'], data['R'])
    return loss


# from https://github.com/ErikOrjehag/sfmnet/blob/5b4001b3950937f604bd394ec2bb14199c1c56d7/networks/unsuperpoint.py
def decorrelate(F):
    # Create a correlation matrix of feature vector F than can be
    # used to formulate a decorrelation loss
    f = F.permute(0,2,1)
    mean = f.mean(dim=-1, keepdims=True)
    b = f - mean
    dot = (b.unsqueeze(2) * b.unsqueeze(1)).sum(dim=-1)
    d = torch.sqrt(dot.diagonal(dim1=1,dim2=2))
    dd = d.unsqueeze(2) * d.unsqueeze(1)
    R = dot / dd
    idx = torch.arange(0,R.shape[1],out=torch.LongTensor())
    R[:,idx,idx] = 0
    return R**2

def uniform_distribution_loss(values, a=0., b=1.):
    # Create a loss that enforces uniform distribution 
    # of values in the interval [a, b]
    v = torch.sort(values.flatten())[0]
    L = v.shape[0]
    i = torch.arange(1, L+1, dtype=torch.float).to(values.device)
    s = ( (v-a) / (b-a) - (i-1) / (L-1) )**2
    return s

def Unsuper_loss(data):
    APrel = data["A"]["Prel"]
    AP = data["A"]["P"]
    AS = data["A"]["S"]
    AF = data["A"]["F"]
    BPrel = data["B"]["Prel"]
    BP = data["B"]["P"]
    BS = data["B"]["S"]
    BF = data["B"]["F"]
    homog = data["homography"]
    H, W = data["img"].shape[2:]

    #print("homog", homog[0], "AP", AP.shape)

    B, N = AS.shape

    T = torch.tensor([
        [1, 0, W/2.],
        [0, 1, H/2.],
        [0, 0, 1],
    ], dtype=torch.float).repeat(B,1,1).to(homog.device)
    
    # Points from branch A transformed by homography        
    APh = from_homog_coords(T @ homog @ torch.inverse(T) @ to_homog_coords(AP))

    # Matrix of distances between points
    D = (APh.permute(0,2,1).unsqueeze(2) - BP.permute(0,2,1).unsqueeze(1)).norm(dim=-1)

    # Create ids which maps the B points to its closest A point (A[ids] <-> B)
    Dmin, ids = torch.min(D, dim=1)

    # Create a mask for only the maped ids that are closer than a threshold.
    mask = Dmin.le(4)


    d = Dmin[mask]
    dmean = d.mean()

    # Distances between corresponding points should be small
    l_position = d

    mask_ = mask.view(-1)
    ids_ = ids.view(-1)
    AS_ = AS.view(-1)[ids_][mask_] # see debugger_point
    BS_ = BS.view(-1)[mask_]

    # Scores of corresonding points should be similar to each other
    l_score = (AS_ - BS_) ** 2

    # Increase score if they are near, supress score if they are far
    S_ = (AS_ + BS_) / 2
    l_usp = S_ * (d - dmean)

    # Descriptor
    C = D.le(8)
    lam_d = 100
    mp = 1
    mn = 0.2
    af = AF.permute(0,2,1).unsqueeze(2)
    bf = BF.permute(0,2,1).unsqueeze(1)
    dot = (af * bf).sum(dim=-1) # [B,N,N]
    pos = torch.clamp(mp - dot, min=0)
    neg = torch.clamp(dot - mn, min=0)
    l_desc = (lam_d * C * pos + (~C) * neg)

    # Decorrelation
    l_decorr_a = decorrelate(AF)
    l_decorr_b = decorrelate(BF)
    l_decorr = 0.5 * (l_decorr_a + l_decorr_b)

    # Uniform distribution of relative positions
    l_uni_ax = uniform_distribution_loss(APrel[:,0,:])
    l_uni_ay = uniform_distribution_loss(APrel[:,1,:])
    l_uni_bx = uniform_distribution_loss(BPrel[:,0,:])
    l_uni_by = uniform_distribution_loss(BPrel[:,1,:])

    # Loss terms
    loss_usp = 1 * l_position.sum() + 2 * l_score.sum() + l_usp.sum()
    loss_desc = l_desc.sum()
    loss_decorr = l_decorr.sum()
    loss_uni_xy = l_uni_ax.sum() + l_uni_ay.sum() + l_uni_bx.sum() + l_uni_by.sum()

    loss = 1 * loss_usp + 0.001 * loss_desc + 0.03 * loss_decorr + 100 * loss_uni_xy

    return loss