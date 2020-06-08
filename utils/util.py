import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

# https://github.com/ErikOrjehag/sfmnet/blob/master/utils.py
import torch
import numpy as np
import random


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


# from https://github.com/ErikOrjehag/sfmnet/blob/master/utils.py

def iterate_loader(device, loader, fn, start=0, end=None, args=[]):
    for step, inputs in enumerate(loader, start=start):
        if end is not None and step >= end:
            break
        inputs = dict_to_device(inputs, device)
        fn(step, inputs, *args)

def forward_pass(model, loss_fn, inputs):
    outputs = model(inputs)
    data = { **inputs, **outputs }
    loss, debug = loss_fn(data)
    data = { **data, **debug }
    return loss, data

def backward_pass(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def dict_to_device(keyval, device):
    return map_dict(keyval, lambda key, val: val.to(device))

def dict_tensors_to_num(keyval):
    return map_dict(keyval, lambda key, val: val.cpu().item())

def dict_append(d1, d2):
    for key, val in d2.items():
        if key not in d1:
            d1[key] = [ val ]
    def fun(key, val):
        if isinstance(val, list):
            return val + [ d2[key] ]
        else:
            return [ val ] + [ d2[key] ]
    return map_dict(d1, fun)

def dict_mean(d):
    return map_dict(d, lambda key, val: np.mean(val))

def dict_std(d):
    return map_dict(d, lambda key, val: np.std(val))

def map_dict(keyval, f):
    return { key: f(key, val) for key, val in keyval.items() }

def normalize_map(map):
    mean_map = map.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    norm_map = map / (mean_map + 1e-7)
    return norm_map

def randn_like(tensor):
    return torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device)

def sigmoid_to_disp_depth(sigmoid, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    disp = min_disp + (max_disp - min_disp) * sigmoid
    depth = 1 / disp
    return disp, depth

def sec_to_hms(t):
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return "{:02d}h{:02d}m{:02d}s".format(t, m, s)

def sum_to_dict(target, source):
    for key, val in source.items():
        if key in target:
            target[key] += val
        else:
            target[key] = val

def is_interval(step, interval):
  return step != 0 and step % interval == 0

def normalize_image(img):
    #x = (input_image - 0.45) / 0.225
    # (img - 0.5) * 0.225 from unsuperpoint paper
    return img * 2 - 1

def cv2_to_torch(img):
    return torch.tensor(np.transpose(np.array(img).astype(np.float32) / 255, axes=(2, 0, 1)))

def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


# mine
# def get_descriptor_by_pos(P, F, image_shape):
#     """obtain descriptors by sampling the appropriate location(P) in the dense descriptor map(F)

#     Args:
#         P (pytorch tensor shaped B x 2 x N): point absolute coordinates
#         F (pytorch tensor shaped B x 256 x 4N): descriptor map
#         img (pytorch tensor shaped B x 3 x H x W): input image data
#     Returns:
#         descriptor (pytorch tensor shaped B x 256 x N): the associated descriptor of key points
#         idx_ (pytorch tensor shaped B x N): the descriptor id in the descriptor map(F)
#     """
#     # TODO - to GPU
#     B = P.shape[0]
#     image_width = image_shape[-1]

#     # # map point to cell
#     P = (P // 4).long()
#     m = torch.tensor([1, image_width/4]).repeat(B,1,1).long()
#     idx = torch.bmm(m, P)              # B x 1 x N tensor
#     idx_ = idx.view(idx.shape[0], -1)   # B x N vector

#     descriptor = torch.stack([F[i,:,idx_[i]] for i in range(idx_.shape[0])], dim=0) # B x 256 x N

#     return descriptor, idx_

# def get_descriptor_by_idx(F, idx):
#     """obtain descriptors by sampling the dense descriptor map(F) by index(idx)

#     Args:
#         F (tensor shaped B x 256 x 4N): [description]
#         idx (tensor shaped B x N): 
#     Returns:
#         descriptor (pytorch tensor shaped B x 256 x N): the associated descriptor of key points
#     """    
#     descriptor = torch.stack([F[i,:,idx[i]] for i in range(idx.shape[0])], dim=0)   # B x 256 x N
#     return descriptor


def brute_force_match(AF, BF):
    # Brute force match descriptor vectors [B,256,N]
    af = AF.permute(0,2,1).unsqueeze(2)
    bf = BF.permute(0,2,1).unsqueeze(1)
    l2 = (af - bf).norm(dim=-1) # [B,N,N]
    dists, Bids = torch.min(l2, dim=2) # [B,N]
    return dists, Bids


def discard_outside_points(P, img_size):
    """discard points whose coordinate is out of image size

    Args:
        P (tensor shaped 2 x N): [description]
        img_size (tuple 3 x H x W): [description]
    """    
    _, H, W = img_size

    m1 = P[0,:].le(W-0.01)
    m2 = P[1,:].le(H-0.01)
    mask = m1 & m2

    P = P.masked_select(mask).reshape(2,-1)
    return P, mask


def get_descriptor_by_pos(P, F, image_size):
    """obtain descriptors by sampling the appropriate location(P) in the dense descriptor map(F)

    Args:
        P (pytorch tensor shaped 2 x M): point absolute coordinates
        F (pytorch tensor shaped 256 x 4N): descriptor map
        img (tuple [3, H, W]): input image data
    Returns:
        descriptor (pytorch tensor shaped 256 x M): the associated descriptor of key points
        idx_ (vector shaped 1xM): the descriptor id in the descriptor map(F)
    """
   
    W = image_size[-1]

    # # map point to cell
    P = (P // 4).long()
    m = torch.tensor([1, W/4]).long()
    idx = m @ P                         # 1xN vector
    descriptor = F[:,idx]               # 256 x N tensor

    return descriptor, idx  # 256xN tensor, 1xN vector

def get_descriptor_by_pos_batch(P, F, image_shape):
    """obtain descriptors by sampling the appropriate location(P) in the dense descriptor map(F)

    Args:
        P (pytorch tensor shaped B x 2 x N): point absolute coordinates
        F (pytorch tensor shaped B x 256 x 4N): descriptor map
        img (pytorch tensor shaped B x 3 x H x W): input image data
    Returns:
        descriptor (pytorch tensor shaped B x 256 x N): the associated descriptor of key points
        idx_ (pytorch tensor shaped B x N): the descriptor id in the descriptor map(F)
    """
    # TODO - to GPU
    B = P.shape[0]
    image_width = image_shape[-1]

    # # map point to cell
    P = (P // 4).long()
    m = torch.tensor([1, image_width/4]).repeat(B,1,1).long()
    idx = torch.bmm(m, P)              # B x 1 x N tensor
    idx_ = idx.view(idx.shape[0], -1)   # B x N vector

    descriptor = torch.stack([F[i,:,idx_[i]] for i in range(idx_.shape[0])], dim=0) # B x 256 x N

    return descriptor, idx_