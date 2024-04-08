# Diffusion Models for Linear Inverse Problems

Implementation for Master's Thesis "Diffusion Models for Image Restoration".

## Training

For training refer to [improved-diffusion](https://github.com/openai/improved-diffusion).


## Sampling

The Class defined in [sample_class](sample_class.py) contains every sampling Algorithm presented.  
The [script_sample](script_sample.py) provides a script to use the sampling Class.  
Usage examples:

```
python script_sample.py --no-conditioned
python script_sample.py --val_dir 'PATH/TO/VALIDATION/DATASET' -p inpaint -m DC --no-sigma-adjusted
python script_sample.py --val_dir 'PATH/TO/VALIDATION/DATASET' -p superres -m PC -t 0 -M0 -s1 0.1
python script_sample.py --val_dir 'PATH/TO/VALIDATION/DATASET' -p  mri_recon_artificial -m PC -t 10 -M 100 -s1 0.4 -s2 0.04 -f T1_320-11 --acc_fact 4
```
