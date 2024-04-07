import torch
import numpy as np
from lxml import etree

from fastmri.data.subsample import RandomMaskFunc
from fastmri.data import transforms as T
from torchvision.transforms import Resize, InterpolationMode, ToTensor, CenterCrop

from utils import *

to_tensor = ToTensor()
centercrop320 = CenterCrop(320)
resize128 = Resize(128, interpolation=InterpolationMode.BICUBIC, antialias=False)

def fft2(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x), norm="ortho"))

def ifft2(y):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(y), norm="ortho"))

def get_new_y_artificial(ft, acc_factor=4, seed=0):
    # function for PROBLEM == mri_recon_artificial
    y = torch.tensor(ft)
    x0 = ifft2(y)
    x_real = x0.real.abs()
    x_imag = x0.imag.abs()
    x_real = norm_0_1_channelwise(x_real)
    x_real = centercrop320(x_real)
    x_real = 2*x_real - 1
    x_imag = norm_0_1_channelwise(x_imag)
    x_imag = centercrop320(x_imag)
    x_imag = 2*x_imag - 1
    x0 = torch.complex(x_real, x_imag)
    y = fft2(x0)
    y = torch.view_as_real(y)
    mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[acc_factor], seed=seed)
    y_masked, mask, _ = T.apply_mask(y, mask_func)
    y_new = torch.view_as_complex(y_masked)
    # mask = np.load("data/mask_acc5.npy")
    # y_new = y * mask
    return y_new.numpy(), mask.numpy().squeeze()

def get_new_y(ft, acc_factor=4, seed=0):
    # function for PROBLEM == mri_recon_magnitude, mri_recon_complex
    y = torch.tensor(ft)
    x0 = ifft2(y)
    x0 = norm_0_1_channelwise(x0.abs()) * torch.exp(1j * x0.angle())
    x0 = centercrop320(x0)
    y = fft2(x0)
    y = torch.view_as_real(y)
    mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[acc_factor], seed=seed)
    y_masked, mask, _ = T.apply_mask(y, mask_func)
    y_new = torch.view_as_complex(y_masked)
    # mask = np.load("data/mask_acc5.npy")
    # y_new = y * mask
    return y_new.numpy(), mask.numpy().squeeze()

def extract_FOV(file_hf):
    root = etree.fromstring(file_hf['ismrmrd_header'][()])
    xml_header = etree.tostring(root, pretty_print=True).decode()
    s = xml_header.find("<reconSpace>")
    s = xml_header.find("<fieldOfView_mm>",s)
    x_pos1 = xml_header.find("<x>",s)
    x_pos2 = xml_header.find("</x>",s)
    y_pos1 = xml_header.find("<y>",s)
    y_pos2 = xml_header.find("</y>",s)
    fov_h = float(xml_header[x_pos1+3:x_pos2])
    fov_w = float(xml_header[y_pos1+3:y_pos2])
    return np.array([fov_h,fov_w])

def extract_matrix_size(file_hf):
    root = etree.fromstring(file_hf['ismrmrd_header'][()])
    xml_header = etree.tostring(root, pretty_print=True).decode()
    s = xml_header.find("<reconSpace>")
    s = xml_header.find("<matrixSize>",s)
    x_pos1 = xml_header.find("<x>",s)
    x_pos2 = xml_header.find("</x>",s)
    y_pos1 = xml_header.find("<y>",s)
    y_pos2 = xml_header.find("</y>",s)
    size_h = float(xml_header[x_pos1+3:x_pos2])
    size_w = float(xml_header[y_pos1+3:y_pos2])
    return np.array([size_h,size_w])

def extract_kspace_size(file_hf):
    root = etree.fromstring(file_hf['ismrmrd_header'][()])
    xml_header = etree.tostring(root, pretty_print=True).decode()
    s = xml_header.find("<encodedSpace>")
    s = xml_header.find("<matrixSize>",s)
    x_pos1 = xml_header.find("<x>",s)
    x_pos2 = xml_header.find("</x>",s)
    y_pos1 = xml_header.find("<y>",s)
    y_pos2 = xml_header.find("</y>",s)
    size_h = float(xml_header[x_pos1+3:x_pos2])
    size_w = float(xml_header[y_pos1+3:y_pos2])
    return np.array([size_h,size_w])

def extract_pixel_size(file_hf):
    fov = extract_FOV(file_hf)
    size = extract_matrix_size(file_hf)
    return fov/size