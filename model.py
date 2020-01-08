import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from PIL import Image


class fft_conv2d(nn.Module):

    def __init__(self, tile_size = 40, kernel_size = 32, tiling_factor = 4):
        super(fft_conv2d, self).__init__()
        self.tile_size = tile_size
        self.kernel_size = kernel_size
        self.tiling_factor = tiling_factor

        pad_one, pad_two = int(np.ceil((self.tile_size - self.kernel_size)/2)), int(np.floor((self.tile_size - self.kernel_size)//2))
        # self.kernels_lists = [[0, 0, 0 ,0],[0, 0, 0 ,0],[0, 0, 0 ,0],[0, 0 ,0 ,0]]
        # for i in range(4):
        #     for j in range(4):
        #         names['kernels_' + str(i) + str(j)] = nn.Parameter(torch.rand(self.kernel_size, self.kernel_size, 2))
        kernels_lists = [[0, 0, 0 ,0],[0, 0, 0 ,0],[0, 0, 0 ,0],[0, 0 ,0 ,0]]
        
        for i in range(self.tiling_factor):
            for j in range(self.tiling_factor):
                # self.names['kernels_' + str(i) + str(j)] = nn.Parameter(torch.rand(self.kernel_size, self.kernel_size, 2))
                exec('self.kernel_{}{} = nn.Parameter(torch.rand(self.kernel_size, self.kernel_size, 2))'.format(i,j))
                exec('kernels_lists[i][j] = self.kernel_{}{}'.format(i,j))
        self.kernels_lists = kernels_lists
        kernels_pad =[[F.pad(kernel, (0,0,pad_one,pad_two, pad_one,pad_two), 'constant', 0) for kernel in kernels]for kernels in                                    self.kernels_lists]
        self.psf_pad = torch.cat([torch.cat(kernel_list, dim=0) for kernel_list in kernels_pad], dim=1)
        self.fc1 = nn.Linear(32, 10)

        

    def forward(self, x):
        x = self.fftconv2d(x, self.psf_pad, otf=None, adjoint=False, phase=True)
        x = self.MaxPool2d_complex(x)
        x = x.view(-1, 4 * 8)
        x = self.fc1(x)
        # nn.MaxPool2d(40, 40)
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        return x
    def MaxPool2d_complex(self, a_tensor):
        # shape = (1,1,160,160,2)
        a_r, a_j = a_tensor.split(1,-1)
        a_r = a_r.squeeze(-1)
        a_j = a_j.squeeze(-1)
        pool = nn.MaxPool2d(40)
        a_r_p = pool(a_r)
        a_j_p= pool(a_j)
        a_p = torch.stack([a_r_p, a_j_p], dim = -1)
        
        return a_p
        
    def torch_conj(self, a_tensor):

        # 复共轭
        otf_conj = torch.cat([a_tensor.index_select(-1, torch.tensor([0])), a_tensor.index_select(-1, torch.tensor([1])) * -1],dim = -1)
        return otf_conj

    def shift2d(self,a_tensor):

        x_shift = (np.shape(a_tensor)[0]+1)//2
        y_shift = (np.shape(a_tensor)[1]+1)//2
        shift2d = torch.roll(a_tensor,shifts = (x_shift, y_shift), dims = (0,1))
        return shift2d

    def ishift2d(self, a_tensor):

        x_shift = (np.shape(a_tensor)[0]+1)//2
        y_shift = (np.shape(a_tensor)[1]+1)//2
        ishift2d = torch.roll(a_tensor,shifts = (-x_shift, -y_shift), dims = (0,1))
        return ishift2d

    def psf2otf(self, psf, psf_size):

        # psf = F.pad()
        psf = self.ishift2d(psf)
        otf = torch.fft(psf, signal_ndim = 2)
        return otf

    def fftconv2d(self, img, psf_pad, otf=None, adjoint=False, phase=True):
        # real2complex
        # if len(img.shape) == 4:
        if img.shape[-1] != 2:
            img_j = torch.zeros_like(img)
            img = torch.stack([img, img_j], dim=-1)

        pad_one, pad_two = int(np.ceil((self.tile_size * self.tiling_factor  - img.shape[-2])/2)), int(np.floor((self.tile_size * self.tiling_factor - img.shape[-3])//2))
        img_pad = F.pad(img,(0, 0, pad_one, pad_two, pad_one, pad_two), "constant", 0)          
        
        img_pad_fft2d = torch.fft(img_pad, signal_ndim = 2)
        
        otf = self.psf2otf(psf_pad, psf_size = self.tile_size * self.tiling_factor)
        otf = otf.unsqueeze(0)
        
        if adjoint:
            result = torch.ifft(img_pad_fft2d * self.torch_conj(otf), signal_ndim = 2)
        else:
            result = torch.ifft(img_pad_fft2d * otf, signal_ndim = 2)

        if phase:  
            result = result
        else:
            result = index_select(-1, torch.tensor([0]))
        return result

