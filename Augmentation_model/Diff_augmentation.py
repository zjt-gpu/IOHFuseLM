import torch
import torch.nn as nn
from Augmentation_model.backbone import Diffusion

class Diff_augmentation(nn.Module):
    
    def __init__(self, configs, device):
        super(Diff_augmentation, self).__init__()
        self.seq_len = configs.seq_len
        
        self.dropout = configs.dropout
        self.aug_time_steps = configs.aug_time_steps
        
        self.diffusion = Diffusion(
            time_steps=configs.aug_time_steps,
            num_features=1,
            seq_len=configs.seq_len,
            device=configs.device
        )
        self.eta = 0
    
    def pred(self, x_inp):
        B, _, N = x_inp.size()
        t = torch.randint(0, self.aug_time_steps, (B,), device=x_inp.device) 
        noise_x = self.diffusion(x_inp, t) 
        x_pred = self.diffusion.pred(noise_x, t)
        
        return x_pred
    

    def forecast(self, x_inp):
        B, _, D = x_inp.shape
        
        shape = torch.zeros((B, self.seq_len, D), dtype=torch.int, device=x_inp.device)
        x_pred = self.diffusion.fast_sample_infill(shape, self.aug_time_steps)
        
        return x_pred

    def forward(self, x, task):
        if task == "train":
            return self.pred(x)  
        elif task == 'valid' or task == "test":
            return self.forecast(x)  
        else:
            raise ValueError(f"Invalid task: {task=}")


