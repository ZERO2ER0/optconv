import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from PIL import Image

class AA(nn.Module):
    def  __init__(self):
        nn.Module.__init__(self)
        self.l1 = a()
        self.l2 = a() 
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class a(nn.Module):
    def  __init__(self):
        nn.Module.__init__(self)
        self.psf = 3
    def forward(self, x):
        x = self.a_add(x) 
        return x 
    def a_add(self,x):
        return x + self.psf

A_A = AA()
print(A_A(3))