import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode, ToTensor, CenterCrop

from utils_mri import fft2, ifft2, get_new_y, get_new_y_artificial
from utils import image_to_tensor, norm_0_1

to_tensor = ToTensor()

def get_condition(PROBLEM, x_or_y, device="cuda:0", acc_factor=4):

    assert PROBLEM in ["inpaint", "superres", "mri_recon_artificial", "mri_recon_magnitude", "mri_recon_complex"], "Problem not implemented."

    if PROBLEM == "inpaint":
        # x_or_y is a numpy array with shape [H,W]
        x = torch.tensor(x_or_y).to(device)
        x = norm_0_1(x)
        mask = image_to_tensor("data/inpaint_mask_320.png").to(device)
        def A(x):
            return x * mask
        At = A
        y = A(x)
        
    elif PROBLEM == "superres":
        # x_or_y is a numpy array with shape [H,W]
        x = torch.tensor(x_or_y).to(device)
        x = norm_0_1(x)
        image_size = x.shape[0]
        if len(x.shape)==2:
            x = x.view(1,1,image_size,image_size)
        downscale = Resize(image_size//2, interpolation=InterpolationMode.BILINEAR, antialias=False)
        upscale = Resize(image_size, interpolation=InterpolationMode.NEAREST)
        def A(x):
            return downscale(x)
        def At(x):
            x = 1/4*upscale(x)
            return x
        y = A(x).squeeze()

    elif "mri_recon" in PROBLEM:
        # x_or_y is a numpy array with shape [H,W] or [C,H,W]
        if "artificial" in PROBLEM:
            y, mask = get_new_y_artificial(x_or_y, acc_factor=acc_factor, seed=0)
        else:
            y, mask = get_new_y(x_or_y, acc_factor=acc_factor, seed=0)
        y = torch.tensor(y, device=device)
        mask = torch.tensor(mask, device=device)
        def A(x):
            ft = fft2(x)
            ft = ft * mask
            return ft
        def At(ft):
            ft = ft * mask
            x = ifft2(ft)
            return x

    return y, A, At