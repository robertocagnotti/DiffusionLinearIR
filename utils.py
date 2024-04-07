import torch
import numpy as np
from PIL import Image

def image_to_tensor(img_path):
    '''
    input: path of image
    output: Tensor of shape (1,k_out,xdim,ydim) with values in [0,1],
            where k_out = 1 or 3 (gray-scale, RGB).
    '''
    img_pil = Image.open(img_path)
    img_pil = img_pil.convert('L')
    ar = np.array(img_pil)
    ar = ar[None, ...]
    img_np = ar.astype(np.float32) / 255.
    img_tens = torch.from_numpy(img_np)
    return img_tens

def image_to_numpy(img_path):
    '''
    input: path of image
    output: Tensor of shape (1,k_out,xdim,ydim) with values in [0,1],
            where k_out = 1 or 3 (gray-scale, RGB).
    '''
    img_pil = Image.open(img_path)
    img_pil = img_pil.convert('L')
    ar = np.array(img_pil)
    img_np = ar.astype(np.float32) / 255.
    return img_np

def norm_0_1(img):
    # normalize any image to [0,1]
    return (img-img.min())/(img.max()-img.min())

def norm_0_1_channelwise(img):
    # input: tensor of size [C,H,W]
    # output: channel wise minimum of size [C]
    assert torch.is_tensor(img), "Input must be torch.Tensor"
    img_min, _ = img.min(dim=1)
    img_min, _ = img_min.min(dim=1)
    img_max, _ = img.max(dim=1)
    img_max, _ = img_max.max(dim=1)
    img_min = img_min.view(-1,1,1)
    img_max = img_max.view(-1,1,1)
    return (img-img_min)/(img_max-img_min)
    
def norm_min1_plus1(img):
    # normalize any image to [-1,1]
    return 2*(img-img.min())/(img.max()-img.min()) - 1

def transform_min1_plus1(img):
    # transform from [0,1] to [-1,1]
    return img/0.5 - 1.0