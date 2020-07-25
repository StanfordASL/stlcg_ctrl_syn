import sys
# sys.path.append('../../../stlcg_karen/src')
# import stlcg
import matplotlib.pyplot as plt

# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import scipy.io as sio
sys.path.append('../')

from torch_interpolations.torch_interpolations.multilinear import RegularGridInterpolator

class ExpertDemoDataset(torch.utils.data.Dataset):

    def __init__(self, npy_file):
        
        # [t, x, y, psi, V]
        self.data = np.load(npy_file)


    def __len__(self):
        return data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        time = self.data[idx, 0]
        state = self.data[idx, 1:5]
        control = self.data[idx, 5:7]
        return {'state': state, 'control': control, 'time': time}




value = sio.loadmat('../../hji/data/value.mat');
deriv_value = sio.loadmat('../../hji/data/deriv_value.mat');
V = value['data'];
dV = [deriv_value['derivC'][i][0] for i in range(4)];
g = sio.loadmat('../../hji/data/grid.mat')['grid'];



values = torch.tensor(V[:,:,:,:,-1]).float()
points = [torch.from_numpy(g[i][0].flatten()).float() for i in range(4)]
value_interp = RegularGridInterpolator(points, values)
deriv_interp = [RegularGridInterpolator(points, torch.tensor(dV[i][:,:,:,:,-1]).float()) for i in range(4)]


class HJIValueFunction(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, input):
        """
        [bs, x_dim]
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return value_interp(input.split(1, dim=1))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        points = input.split(1, dim=1)
        gr = torch.zeros_like(input)
        return  torch.cat([deriv_interp[i](points) for i in range(4)], 1) * grad_output


if __name__ == '__main__':
    vf = HJIValueFunction.apply
    inputs = torch.stack([p[:3] for p in points], 1).float().requires_grad_(True)
    loss = vf(inputs[1:2,:]).squeeze()
    loss.backward()
    print(inputs.grad)
    print(torch.stack([deriv_interp[i]([p[:3] for p in points]) for i in range(4)], 1))