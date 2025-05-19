import torch
import torch.nn as nn
from Augmentation_model.diffts import diffts
from functools import partial

class Moving_MultiAvg(nn.Module):

    def __init__(self, kernel_sizes, stride):
        super(Moving_MultiAvg, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.avg_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=stride, padding=0) for k in kernel_sizes
        ])

    def forward(self, x):
        smoothed = []
        for k, avg in zip(self.kernel_sizes, self.avg_pools):
            front = x[:, 0:1, :].repeat(1, (k - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (k - 1) // 2, 1)
            x_padded = torch.cat([front, x, end], dim=1)

            smoothed_x = avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
            smoothed.append(smoothed_x)
        return torch.mean(torch.stack(smoothed), dim=0)

class series_decomp(nn.Module):

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = Moving_MultiAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.to(a.device)  
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class Diffusion(nn.Module):
    def __init__(
        self,
        time_steps: int,
        num_features : int,
        seq_len : int,
        device : torch.device
    ):
        super(Diffusion, self).__init__()
        self.time_steps = time_steps
        self.seq_length = seq_len
        self.device = device

        self.betas = self._cosine_beta_schedule().to(self.device)
        
        self.eta = 0
        self.alpha = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)
        
        self.diff = diffts(num_features, seq_len, self.device)
        

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1).unsqueeze(-1).to(x.device)
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x, t):
        noisy_x, _ = self.noise(x, t)
        return noisy_x
    
    def pred(self, x, t):
        if t == None:
            t = torch.randint(0, self.time_steps, (x.shape[0],), device=self.device)
        return self.diff(x, t)
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, t):
        
        maybe_clip = partial(torch.clamp, min=-1., max=1.)
        x_start = self.diff(x, t)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start
        
    @torch.no_grad()
    def fast_sample_infill(self, shape, sampling_timesteps):
        batch_size, _, _ = shape.shape
        batch, device, total_timesteps, eta = shape[0], self.device, self.time_steps, self.eta
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 
        shape = shape
        img = torch.randn(shape.shape, device=device)
        
        for time, time_next in time_pairs:
            time = torch.full((1,), time, device=device, dtype=torch.long)
            
            pred_noise, x_start, *_ = self.model_predictions(img, time)

            if time_next < 0:
                img = x_start
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise

        return img
