import torch
import numpy as np
import os
import h5py
import argparse

from sample_class import SAMPLE
from improved_ddpm.script_util import create_model
from utils_inverse_problems import get_condition
from utils_mri import extract_pixel_size

image_size=320
device="cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument('--val_dir', type=str)
parser.add_argument('--conditioned', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('-p', '--problem', type=str)
parser.add_argument('-m', '--sampling_mode', type=str, default="DC")
parser.add_argument('--sigma_adjusted', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('-s1', '--sigma1', type=float, default=1)
parser.add_argument('-s2', '--sigma2', type=float, default=1)
parser.add_argument('-t', '--corrector_steps', type=int, default=0)
parser.add_argument('-M', '--M', type=int, default=0)
parser.add_argument('-n', '--num_samples', type=int, default=1)
parser.add_argument('-d', '--dir_num', type=int, default=0)
parser.add_argument('-s', '--slice', type=int, default=0)
parser.add_argument('-f', '--file', type=str, default="T1_320_01")
parser.add_argument('-a','--acc_factor', type=int, default=4)
args = parser.parse_args()

dir_main = args.val_dir
dirs = sorted(os.listdir(dir_main))

### Only include files with "T1" in their name
types = ["T1"]
if types != None:
    dirs = [dir for dir in dirs for type in types if type in dir ]
dirs = [os.path.join(dir_main,dir) for dir in dirs]

### Uncomment the following line to only includes volumes with the same pixel size
#dirs = [dir for dir in dirs if tuple(extract_pixel_size(h5py.File(dir)))==(0.6875, 0.6875)]

conditioned = args.conditioned
PROBLEM = args.problem
SAMPLING_MODE = args.sampling_mode
sigma_adjusted = args.sigma_adjusted
sigma1 = args.sigma1
sigma2 = args.sigma2
corrector_time_steps = args.corrector_steps
M = args.M
num_samples = args.num_samples
dir_num = args.dir_num
slice = args.slice
file=args.file
acc_factor=args.acc_factor

if file == "T1_320_01":
    model_name = "ema_0.9999_700000.pt"
elif file == "T1_320_-11":
    model_name = "ema_0.9999_780000.pt"

condition_dir = dirs[dir_num]

if SAMPLING_MODE == "PC":
    sigma_adjusted = True

print("Hyperparameters:")
for arg in vars(args):
    print(arg,":", getattr(args,arg))

if conditioned == True:
    assert PROBLEM in ["inpaint", "superres", "mri_recon_artificial", "mri_recon_magnitude", "mri_recon_complex"], \
        "Problem must be one of (inpaint, superres, mri_recon_artificial, mri_recon_magnitude, mri_recon_complex)"
    
    assert PROBLEM != "mri_recon_complex", "Complex model has not been trained yet"

    if PROBLEM == "mri_recon_magnitude":
        assert SAMPLING_MODE == "DC", "Magnitude model only compatible with DC sampling"

channels=1
if PROBLEM == "mri_recon_complex":
    channels=2

print("creating model...")
model = create_model(
    image_size=image_size,
    channels=channels,
    num_channels=128,
    num_res_blocks=2,
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=True,
    attention_resolutions="16",
    num_heads=1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0.0)

print("loading from checkpoint...")
ema_ckpt = torch.load(f"trained_models/{file}/{model_name}", map_location=device)
model.load_state_dict(ema_ckpt)

if conditioned:
    print("generating condition...")
    hf = h5py.File(condition_dir)
    GT = hf["reconstruction_rss"][()][slice]

    if PROBLEM in ["inpaint", "superres"]:
        x_or_y = GT
        assert x_or_y.shape == (320,320), "Please resize the condition"
    elif "mri_recon" in PROBLEM:
        kspace = hf["kspace"][()]
        x_or_y = kspace[slice]
    y, A, At = get_condition(PROBLEM, x_or_y, device=device, acc_factor=acc_factor)
else:
    PROBLEM = "unconditional"
    y=None; A=None; At=None

os.makedirs(f"outputs/{file}/{PROBLEM}", exist_ok=True)

print("sampling...")
torch.manual_seed(100)
SAMPLE_INSTANCE = SAMPLE(model,
                        image_size=image_size,
                        conditioned=conditioned,
                        PROBLEM=PROBLEM,
                        y=y,
                        A=A,
                        At=At,
                        num_samples=num_samples,
                        device=device)

torch.manual_seed(0)
if SAMPLING_MODE == "PC":
    sample = SAMPLE_INSTANCE.run_PC_sampling(sigma1=sigma1,
                                             sigma2=sigma2,
                                             M=M,
                                             corrector_time_steps=corrector_time_steps)
elif SAMPLING_MODE == "DC":
    sample = SAMPLE_INSTANCE.run_DC_sampling(sigma1=sigma1,
                                             sigma2=sigma2,
                                             M=M,
                                             corrector_time_steps=corrector_time_steps,
                                             sigma_adjusted=sigma_adjusted)

if conditioned:
    save_name = ""
    if "mri_recon" in PROBLEM:
        save_name = f"acc{acc_factor:02d}_"
    if sigma_adjusted:
        save_name += f"dir{dir_num:03d}_slice{slice:02d}_{SAMPLING_MODE}_sigma1_{sigma1:.2f}_sigma2_{sigma2:.2f}_t{corrector_time_steps:03d}_M{M:03d}"
    else:
        save_name += f"dir{dir_num:03d}_slice{slice:02d}_DC_not_sigma_adjusted_t{corrector_time_steps:03d}_M{M:03d}"
    out = np.array([y.cpu().squeeze().numpy(), sample, GT], dtype=object)
else:
    save_name = f"sample"
    out = sample

np.save(f"outputs/{file}/{PROBLEM}/{save_name}.npy", out)
print("sampled!")
