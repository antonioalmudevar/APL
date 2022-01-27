import torch
import torch.nn as nn

class AdaIN(nn.Module):

    def __init__(self):
        super().__init__()

    def mu_3d(self, d):
        return torch.sum(d,(2))/(d.shape[2])

    def sigma_3d(self, d):
        return torch.sqrt((torch.sum((d.permute([2,0,1])-self.mu_3d(d)).permute([1,2,0])**2, 2)+0.000000023)/d.shape[2])

    def mu_4d(self, x):
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma_4d(self, x):
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu_4d(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))
    
    def forward(self, x, d):
        return (self.sigma_3d(d)*((x.permute([2,3,0,1])-self.mu_4d(x))/self.sigma_4d(x)) + self.mu_3d(d)).permute([2,3,0,1])