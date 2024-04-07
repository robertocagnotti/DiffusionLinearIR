import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
from utils import norm_0_1
from utils_mri import ifft2
import torch

def psnr(x_hat,x_true,maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse=np.mean(np.square(x_hat-x_true))
    psnr_ = 10.*np.log(maxv**2/mse)/np.log(10.)
    return psnr_

def plot_samples(main_dir):

    if "unconditional" in main_dir:
        sample = np.load(f"{main_dir}/sample.npy")
        num_samples = sample.shape[0]
        if num_samples == 1:
            plt.imshow(sample[0], cmap="gray")
        else:
            fig,axs = plt.subplots(num_samples, 1, figsize=(5,5*num_samples), layout="constrained")
            for ax in axs.ravel():
                ax.set_axis_off()
            for i in range(sample.shape[0]):
                axs[i].imshow(sample[i], cmap="gray")

    else:
        samples_dirs = sorted(os.listdir(main_dir))
        num_samples = len(samples_dirs)

        fig,axs = plt.subplots(num_samples, 3, figsize=(15,5*num_samples), layout="constrained")

        for ax in axs.ravel():
            ax.set_axis_off()

        if num_samples != 1:
            for i,dir in enumerate(samples_dirs):
                y, sample, GT = np.load(f"{main_dir}/{dir}", allow_pickle=True)
                if np.iscomplex(y).any():
                    x = ifft2(torch.tensor(y)).abs()[0]
                    y = np.abs(y)
                    y[y<1e-10]=1e-10
                    y = np.log(y)
                    if len(y.shape)==3:
                        y = y[1]
                sample = sample.squeeze()
                if np.iscomplex(sample).any():
                    real = (sample.real+1)/2
                    imag = (sample.imag+1)/2
                    sample = np.sqrt(real**2+imag**2).clip(0,1)
                sample = norm_0_1(sample)
                GT = norm_0_1(GT)
                psnr_ = psnr(sample, GT)
                ssim_ = ssim(sample, GT, data_range=1)
                axs[i,0].set_title("Perturbed", fontsize=15)
                axs[i,1].set_title(f"{dir[:-4]} \n PSNR:{psnr_:.2f} SSIM:{ssim_:.2f}", fontsize=15)
                axs[i,2].set_title("GT", fontsize=15)
                axs[i,0].imshow(y, cmap="gray")
                axs[i,1].imshow(sample, cmap="gray")
                axs[i,2].imshow(GT, cmap="gray")
        else:
            dir = samples_dirs[0]
            y, sample, GT = np.load(f"{main_dir}/{dir}", allow_pickle=True)
            if np.iscomplex(y).any():
                x = ifft2(torch.tensor(y)).abs()[0]
                y = np.abs(y)
                y[y<1e-10]=1e-10
                y = np.log(y)
                if len(y.shape)==3:
                    y = y[1]
            sample = sample.squeeze()
            if np.iscomplex(sample).any():
                real = (sample.real+1)/2
                imag = (sample.imag+1)/2
                sample = np.sqrt(real**2+imag**2).clip(0,1)
            sample = norm_0_1(sample)
            GT = norm_0_1(GT)
            psnr_ = psnr(sample, GT)
            ssim_ = ssim(sample, GT, data_range=1)
            axs[0].set_title("Perturbed", fontsize=15)
            axs[1].set_title(f"{dir[:-4]} \n PSNR:{psnr_:.2f} SSIM:{ssim_:.2f}", fontsize=15)
            axs[2].set_title("GT", fontsize=15)
            axs[0].imshow(y, cmap="gray")
            axs[1].imshow(sample, cmap="gray")
            axs[2].imshow(GT, cmap="gray")