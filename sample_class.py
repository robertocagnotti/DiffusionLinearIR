import numpy as np
import torch
from torchvision.transforms import ToTensor
from utils_mri import ifft2

to_tensor = ToTensor()

class SAMPLE:
    def __init__(self, 
                model,
                image_size=320,
                x_start=None,
                conditioned=False,
                PROBLEM=None,
                y=None,
                A=None,
                At=None,
                num_samples=1,
                time_steps=1000,
                b_min=1e-4, 
                b_max=2e-2, 
                device="cuda:0"):
        
        self.image_size = image_size
        self.model = model
        self.conditioned = conditioned
        self.num_samples = num_samples
        self.time_steps = time_steps
        db = (b_max-b_min)/(time_steps-1)
        self.betas = np.arange(start=b_min,stop=b_max,step=db)
        self.alphas = (1 - self.betas)
        self.alphas_bar = np.array([np.prod(self.alphas[:t]) for t in range(1,time_steps+1)])
        self.device = device

        self.y = None
        self.PROBLEM = PROBLEM
        self.A = A
        self.At = At

        if "mri_recon" in PROBLEM:
            self.dtype = torch.complex64
        else:
            self.dtype = torch.float32

        if x_start is None:
            if PROBLEM == "mri_recon_artificial":
                real = torch.randn((num_samples,1,image_size,image_size), device=device)
                imag = torch.randn((num_samples,1,image_size,image_size), device=device)
                self.x_start = torch.complex(real, imag)
            elif PROBLEM == "mri_recon_complex":
                self.x_start = torch.randn((num_samples,2,image_size,image_size), device=device)
            else:
                self.x_start = torch.randn((num_samples,1,image_size,image_size), device=device)
        else:
            self.x_start = torch.unsqueeze(to_tensor(x_start),0).to(device)
            self.x_start = self.x_start.float()

        if PROBLEM == "mri_recon_magnitude":
            self.mask_asc = torch.zeros(image_size, device=device)
            center_size = int(image_size * 0.04)
            l = image_size//2-center_size//2
            r = image_size//2+center_size//2+1
            self.mask_asc[l:r]=1

        if conditioned:
            if len(y.shape)==2:
                y = y.unsqueeze(0)
            self.y = torch.stack([y]*num_samples).to(self.device)
            
    def sample_yt(self, t):
        if "mri_recon" in self.PROBLEM:
            real = torch.randn((self.num_samples,self.y.shape[1],self.image_size,self.image_size), device=self.y.device)
            imag = torch.randn((self.num_samples,self.y.shape[1],self.image_size,self.image_size), device=self.y.device)
            eps = torch.complex(real, imag)
        else:
            eps = torch.randn((self.num_samples,self.y.shape[1],self.image_size,self.image_size), device=self.y.device)
        return self.y*np.sqrt(self.alphas_bar[t]) + np.sqrt(1-self.alphas_bar[t]) * self.A(eps)

    def score(self, x, ts, yt=None, sigma=1):
        t = int(ts[0].cpu())
        prior_score = 0
        likelihood_score = 0
        if self.PROBLEM == "mri_recon_artificial":
            out_real = self.model(x.real,ts).detach()
            out_imag = self.model(x.imag,ts).detach()
            out_complex = torch.complex(out_real, out_imag)
            prior_score = -(out_complex)/np.sqrt(1-self.alphas_bar[t])          
        else:
            out = self.model(x,ts).detach()
            prior_score = -out/np.sqrt(1-self.alphas_bar[t])

        if sigma!=0 and self.conditioned:
            likelihood_score = (-1/sigma**2 * self.At(self.A(x)-yt).sum(dim=1, keepdim=True)).type(self.dtype)
        return prior_score + likelihood_score

    def langevin_step(self, x, ts, sigma=1, yt=None):
        t = int(ts[0].cpu())
        z = torch.randn_like(x, device=x.device)
        score = self.score(x,ts,yt=yt,sigma=sigma)
        r = 0.25
        eps = 2*self.alphas[t]*(r*z.norm()/score.norm())**2
        eps = float(eps)
        out = x + eps * score + np.sqrt(2*eps) * z
        return out

    def ancestral_step(self, x, ts, sigma=1, yt=None):
        t = int(ts[0].cpu())
        a = 2-np.sqrt(1-self.betas[t])
        b = self.betas[t]
        c = np.sqrt(self.betas[t])
        score = self.score(x,ts,yt=yt,sigma=sigma)
        if self.PROBLEM == "mri_recon_artificial":
            real = torch.randn(x.shape, device=x.device)
            imag = torch.randn(x.shape, device=x.device)
            z = torch.complex(real, imag)
        else:
            z = torch.randn_like(x, device=x.device)
        return a * x + b * score + c * z
    
    def data_consistency_step(self, x, ts, sigma=1, yt=None, sigma_adjusted=True):
        t = int(ts[0].cpu())

        if self.PROBLEM == "mri_recon_complex":
            x = torch.complex(x[:,0,:,:],x[:,1,:,:])
        elif self.PROBLEM == "mri_recon_magnitude":
            phi = self.get_angle(yt)
            x = x * torch.exp(1j*phi)
        
        if sigma_adjusted:
            c = self.betas[t]/sigma**2
            # c = 1/sigma**2
        else:
            c = 1

        if self.PROBLEM != "mri_recon_magnitude":
            x = (x + c * self.At(yt - self.A(x)).sum(dim=1, keepdim=True)).type(self.dtype)
        else:
            x = (x + c * self.At(yt - self.A(x))).type(self.dtype)

        if self.PROBLEM == "mri_recon_complex":
            x = torch.view_as_real(x[0]).permute(0,3,1,2)
        elif self.PROBLEM == "mri_recon_magnitude":
            x = x.abs().mean(dim=1, keepdim=True)
        return x
    
    def get_angle(self, y):
        y_asc = y * self.mask_asc
        x_asc = ifft2(y_asc)
        phi = x_asc.angle().to(self.device)
        return phi
    

    # PREDICTOR CORRECTOR
    def run_PC_sampling(self, sigma1=1, sigma2=1, M=0, corrector_time_steps=1):
        
        self.model.to(self.device)
        self.model.eval()

        predictor_step = self.ancestral_step
        corrector_step = self.langevin_step

        x = self.x_start

        for t in range(self.time_steps-1, -1, -1):
            if t%50==0 or t<corrector_time_steps:
                print(f"Timestep: {t:03d}", end='\r')
            ts = torch.tensor([t]*self.num_samples, device=self.device)
            if self.conditioned:
                yt = self.sample_yt(t)
            else:
                yt = None
            x = predictor_step(x, ts, sigma=sigma1, yt=yt)
            if t<corrector_time_steps:
                for _ in range(M):
                    x = corrector_step(x, ts, sigma=sigma2, yt=yt)
        
        out = x.data.cpu().numpy().transpose(0,2,3,1)
        return out

    # DATA CONSISTENCY
    def run_DC_sampling(self, sigma1=1, sigma2=1, M=0, corrector_time_steps=1, sigma_adjusted=True):

        self.model.to(self.device)
        self.model.eval()

        predictor_step = self.ancestral_step
        corrector_step = self.langevin_step
        # corrector_step = self.ancestral_step
        DC_step = self.data_consistency_step

        x = self.x_start

        for t in range(self.time_steps-1, -1, -1):
            if t%50==0 or t<corrector_time_steps:
                print(f"Timestep: {t:03d}", end='\r')
            ts = torch.tensor([t]*self.num_samples, device=self.device)
            if self.conditioned:
                yt = self.sample_yt(t)
            else:
                yt = None
            x = predictor_step(x, ts, sigma=0)
            if self.conditioned:
                x = DC_step(x, ts, sigma=sigma1, yt=yt, sigma_adjusted=sigma_adjusted)
            if t<corrector_time_steps:
                for _ in range(M):
                    x = corrector_step(x, ts, sigma=0)
                    if self.conditioned:
                        x = DC_step(x, ts, sigma=sigma2, yt=yt, sigma_adjusted=sigma_adjusted)
        
        out = x.data.cpu().numpy().transpose(0,2,3,1)
        return out