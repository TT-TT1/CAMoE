import torch
import torch.nn as nn
import torch.nn.functional as F

class router(nn.Module):
    def __init__(self, dim, channel_num, init_temp=1.0, min_temp=0.1, max_temp=10.0):

        super().__init__()
        self.l1 = nn.Linear(dim, int(dim/8))
        self.l2 = nn.Linear(int(dim/8), channel_num)
        
    
        self.temp_net = nn.Sequential(
            nn.Linear(dim, dim//4), 
            nn.ReLU(),
            nn.Linear(dim//4, 1),   
            nn.Sigmoid()            
        )

        self.min_temp = min_temp
        self.max_temp = max_temp
        self.init_temp = init_temp

        self.temperature_cache = None
        
    def forward(self, x):

        x_flat = x.view(x.shape[0], -1)

        temp_ratio = self.temp_net(x_flat)

        temperature = self.min_temp + temp_ratio * (self.max_temp - self.min_temp)

        self.temperature_cache = temperature.detach()

        hidden = F.relu(F.normalize(self.l1(x_flat), p=2, dim=1))
        logits = self.l2(hidden)

        scaled_logits = logits / temperature  
        
       
        output = torch.softmax(scaled_logits, dim=1)
        
        return output
    
    def get_temperature_stats(self):
        
        if self.temperature_cache is not None:
            return {
                'mean': float(self.temperature_cache.mean().item()),
                'std': float(self.temperature_cache.std().item()),
                'min': float(self.temperature_cache.min().item()),
                'max': float(self.temperature_cache.max().item())
            }
        return None
