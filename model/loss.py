import torch
import torch.nn.functional as F

import base

from utils import from_homog_coords, to_homog_coords, get_descriptor_by_pos, discard_outside_points

# demo loss
def nll_loss(output, target):
    return F.nll_loss(output, target)

def kpn_loss(A, B, homog): 
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
        ], dtype=torch.float).repeat(B,1,1).to(homog.device)
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


    """
    # get descriptor of source key points
    fs, _ = get_descriptor_by_pos(AP, AF, img_size)       # B x 256 x N tensor, key point's descriptor in source image
    fpos, fpos_idx = get_descriptor_by_pos(APh, BF, img_size)     # B x 256 x N , B x N
    fpos_norm = (fs-fpos).norm(dim=1)   # B x N

    # Find the cloest descriptor of 'fs' in target descriptor map(BF)
    FD = (fs.permute(0,2,1).unsqueeze(2) - BF.permute(0,2,1).unsqueeze(1)).norm(dim=-1) # B x N x 4N

    # get index of top two cloest descriptor(potential negative sample)
    _, fneg_idxs = torch.topk(FD, 2, largest=False, sorted=False)  # B x N x 2, B x N x 2
    mask = fneg_idxs[:,:,0]==fpos_idx # B x N tensor
    mask = torch.stack([mask,~mask],dim=-1) # B x N x 2 tensor
    # get index of negative sample
    fneg_idx = fneg_idxs[mask].reshape(mask[:,:,0].shape)  # B x N vector -> BxN tensor

    fneg = get_descriptor_by_idx(BF, fneg_idx) # B x 256 x N
    fneg_norm = (fs-fneg).norm(dim=1)   # B x N tensor

    # distance margin parameter
    m = 0.2
    l_dsp = torch.max(torch.zeros_like(fpos_norm), fpos_norm-fneg_norm + m)

    desc_loss = l_dsp.sum()
    """

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
    # select keypoints with the lowest K predicted scores
    # TODO - move global parameter to config.json
    epsilon_uv = 1.0

    Ps, Pt_star = PP[:, 0:2, :], PP[:, 2:4, :]  # Bx2xN tensor

    tmp = torch.sign((Ps-Pt_star).norm(dim=1).unsqueeze(1)-epsilon_uv) #Bx1xN tensor

    l_io = (R-tmp).norm(dim=1)
    loss_io = 0.5 * l_io.sum()

    return loss_io


def KP2D_loss(data):

    return kpn_loss(data['A'], data['B'], data['H']) + ion_loss(data['PP'], data['R'])
