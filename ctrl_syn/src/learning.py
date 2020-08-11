import sys
sys.path.append('../../../stlcg_karen/src')
sys.path.append('../../expert_demo_ros/src/utils')
sys.path.append('../')
import stlcg
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import scipy.io as sio

from torch_interpolations.torch_interpolations.multilinear import RegularGridInterpolator
from torch.utils.tensorboard import SummaryWriter

from environment import *
import IPython

# value = sio.loadmat('../../hji/data/coverage/value.mat');
# deriv_value = sio.loadmat('../../hji/data/coverage/deriv_value.mat');
# V = value['data'];
# g = sio.loadmat('../../hji/data/coverage/grid.mat')['grid'];


value = sio.loadmat('../../hji/data/reach_goal/value.mat');
deriv_value = sio.loadmat('../../hji/data/reach_goal/deriv_value.mat');
V = value['data'];
g = sio.loadmat('../../hji/data/reach_goal/grid.mat')['grid'];


values = torch.tensor(V[:,:,:,:,-1]).float()
dV = [torch.tensor(deriv_value['derivC'][i][0]).float() for i in range(4)];
points = [torch.from_numpy(g[i][0].flatten()).float() for i in range(4)]

value_interp = RegularGridInterpolator(points, values)
deriv_interp = [RegularGridInterpolator(points, dV[i][:,:,:,:,-1]) for i in range(4)]


def plot_hji_contour():
    proj = np.min(np.min(V[:,:,:,:,-1], -1), -1)
    X, Y = np.meshgrid(g[0][0].flatten(), g[1][0].flatten())
    plt.contourf(X, Y, proj.T, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], alpha=0.5)

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
        contiguous_input = [p.contiguous() for p in input.split(1, dim=-1)]
        return value_interp(contiguous_input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        points = [p.contiguous() for p in input.split(1, dim=-1)]
        dim = input.shape[-1]

        return  torch.cat([deriv_interp[i](points) for i in range(dim)], -1) * grad_output


value_interp_cuda = RegularGridInterpolator([p.cuda() for p in points], values.cuda())
deriv_interp_cuda = [RegularGridInterpolator([p.cuda() for p in points], dV[i][:,:,:,:,-1].cuda()) for i in range(4)]

class HJIValueFunction_cuda(torch.autograd.Function):

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
        contiguous_input = [p.contiguous() for p in input.split(1, dim=-1)]
        return value_interp_cuda(contiguous_input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        points = [p.contiguous() for p in input.split(1, dim=-1)]
        dim = input.shape[-1]

        return  torch.cat([deriv_interp_cuda[i](points) for i in range(dim)], -1) * grad_output







# vf = HJIValueFunction.apply

def save_model(model, optim, epoch, loss, PATH):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss}, PATH)

def prepare_data(npy_file):
    '''
    data is a numpy of size [time_dim, 7]. The columns are: [t, x, y, psi, V, a, delta]
    This extracts the states and controls and turns them into tensors of size [time_dim, 1, state/ctrl_dim]
    This also outputs the mean and std of the data [1, 1, state+ctrl_dim]
    '''
    data = np.load(npy_file)[:,1:]
    μ = np.mean(data, axis=0, keepdims=True)
    σ = np.std(data, axis=0, keepdims=True)
    x = torch.tensor(data[:, :4]).float().unsqueeze(1).requires_grad_(False)
    u = torch.tensor(data[:, 4:6]).float().unsqueeze(1).requires_grad_(False)
    return x, u, [torch.tensor(μ).float().unsqueeze(1).requires_grad_(False), torch.tensor(σ).float().unsqueeze(1).requires_grad_(False),]




def kinematic_bicycle(x, u, dt=0.5, stats=(torch.zeros([1,1,6]), torch.ones([1,1,6])), lr=0.7, lf=0.5, V_min=0.0, V_max=5.0, a_min=-3, a_max=3, delta_min=-0.344, delta_max=0.344):
    '''
    x is [..., state_dim]
    u is [..., ctrl_dim]

    The kinematic bicycle model with zero-order hold on controls.
    Computed using mathematica
    '''
    μ = stats[0]
    σ = stats[1]
    # unscale
    x, y, psi, V = unstandardize_data(x, μ[:,:,:4], σ[:,:,:4]).split(1, dim=-1)
    a, delta = unstandardize_data(u, μ[:,:,4:], σ[:,:,4:]).split(1, dim=-1)

    beta = torch.atan(lr / (lr + lf) * torch.tan(delta.clamp(delta_min, delta_max)))

    tol = 1E-3
    a = a.clamp(a_min, a_max)

    int_V =  torch.where(a == 0,
                         V * dt, 
                         torch.where(a > 0, torch.where(((V_max - V) / a) >= dt, 
                                                        (0.5 * a * dt**2 + dt * V) * torch.sin(beta) / lr, 
                                                        (V * ((V_max - V) / a) + 0.5 * ((V_max - V) / a) * (V_max - V) + (dt - ((V_max - V) / a)) * V_max) * torch.sin(beta) / lr),
                                            torch.where(((V_min - V) / a) >= dt, 
                                                        (0.5 * a * dt**2 + dt * V) * torch.sin(beta) / lr, 
                                                        (0.5 * V * ((V_min - V) / a)) * torch.sin(beta) / lr)
                                    )
                         )

    psi_new = psi + int_V
    V_new = (a*dt + V).clamp(V_min, V_max)
    x_new = x + torch.where(torch.abs(beta) > tol,
                            -lr * torch.sin(beta + psi) / torch.sin(beta) + lr * torch.sin(beta + psi_new) / torch.sin(beta),
                            int_V * torch.cos(psi))
    y_new = y + torch.where(torch.abs(beta) > tol,
                            lr * torch.cos(beta + psi) / torch.sin(beta) - lr * torch.cos(beta + psi_new) / torch.sin(beta),
                            int_V * torch.sin(psi))
    # scale
    return standardize_data(torch.cat([x_new, y_new, psi_new, V_new], dim=-1), μ[:,:,:4], σ[:,:,:4])


def initial_conditions(n, vf):
    '''
    Use rejection sampling to sample n initial states from a box and inside the reachable set of the initial state of the expert demonstration
    returns a [1, n, state_dim] numpy
    '''

    good_samples = np.zeros([1,2*n,4])
    total = 0
    while total < n:
        x0 = np.random.rand(1,n,4)
        x0[:,:,:2] -= 0.5
        x0[:,:,:2] *= 2.0
        x0[:,:,2] *= np.pi / 2
        x0[:,:,2] += np.pi / 4
        x0[:,:,3] *= 2
        p = torch.tensor(x0).float()
        v = vf(p).squeeze().numpy() < 0
        num_new = np.sum(v)
        good_samples[:,total:num_new+total,:] = x0[:,v,:]
        total += num_new
    return good_samples[:,:n,:]


class InitialConditionDataset(torch.utils.data.Dataset):

    def __init__(self, n, vf):
        self.n = n
        self.ic = initial_conditions(n, vf)


    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.ic[:,idx,:]



def standardize_data(x, mu, sigma):
    return (x - mu)/sigma

def unstandardize_data(x, mu, sigma):
    return x * sigma + mu




# def outside_circle_stl(signal, circle):
#     d2 = stlcg.Expression('d2_to_center', (signal[:,:,:2] - torch.tensor(circle.center).unsqueeze(0).unsqueeze(0)).pow(2).sum(-1, keepdim=True))
#     return stlcg.Always(subformula = d2 > circle.radius), d2


# def in_box_stl(signal, box):
#     x = stlcg.Expression('x', signal[:,:,:1])
#     y = stlcg.Expression('y', signal[:,:,1:2])
#     return ((x > box.lower[0]) & (y > box.lower[1])) & ((x < box.upper[0]) & (y < box.upper[1])), ((x, y),(x, y))


    
class STLPolicy(torch.nn.Module):

    def __init__(self, dynamics, state_dim, ctrl_dim, hidden_dim, stats, env, value_func, dropout=0., num_layers=1, dt = 0.5, a_min=-3, a_max=3, delta_min=-0.344, delta_max=0.344):
        super(STLPolicy, self).__init__()
        
        self.dynamics = dynamics
        self.stats = stats
        self.dt = dt      
        self.state_dim = state_dim
        self.env = env
        self.value_func = value_func

        a_lim_ = torch.tensor([a_min, a_max]).float().unsqueeze(0).unsqueeze(0)
        delta_lim_ = torch.tensor([delta_min, delta_max]).float().unsqueeze(0).unsqueeze(0)
        self.a_lim = standardize_data(a_lim_, stats[0][:,:,4:5], stats[1][:,:,4:5])
        self.delta_lim = standardize_data(delta_lim_, stats[0][:,:,5:], stats[1][:,:,5:])
        
        self.lstm = torch.nn.LSTM(state_dim, hidden_dim, num_layers, dropout=dropout)
        self.proj = torch.nn.Sequential(torch.nn.Linear(hidden_dim, ctrl_dim), torch.nn.Tanh())
        self.initialize_rnn_h = torch.nn.Linear(state_dim, hidden_dim)
        self.initialize_rnn_c = torch.nn.Linear(state_dim, hidden_dim)
        self.L2loss = torch.nn.MSELoss()
        

    def switch_device(self, device):
        self.a_lim = self.a_lim.to(device)
        self.delta_lim = self.delta_lim.to(device)
        self.stats[0] = self.stats[0].to(device)
        self.stats[1] = self.stats[1].to(device)

    def initial_rnn_state(self, x0):
        # x0 is [bs, state_dim]
        return self.initialize_rnn_h(x0), self.initialize_rnn_c(x0)
    
    def forward(self, x):        
        # x is [time_dim, bs, state_dim]
        h0 = self.initial_rnn_state(x[:1,:,:])

        o, _ = self.lstm(x, h0)    # [time_dim, bs, hidden_dim] , bs = 1 for a single expert trajectory.
        
        # [time_dim, bs, ctrl_dim]  projecting between u_min and u_max (standardize) since proj is between -1 and 1 due to tanh
        u_ = self.proj(o)    # [a, delta]
        a = (self.a_lim[:,:,1:] - self.a_lim[:,:,:1]) / 2 * u_[:,:,:1] + self.a_lim.mean(-1, keepdims=True)
        delta = (self.delta_lim[:,:,1:] - self.delta_lim[:,:,:1]) / 2 * u_[:,:,1:] + self.delta_lim.mean(-1, keepdims=True)
        u = torch.cat([a, delta], dim=-1)
        x_next = self.dynamics(x, u, stats=self.stats)
        x_pred = self.join_partial_future_signal(x[:1,:,:], x_next[:-1,:,:])
        return o, u, x_pred
            
    
    def propagate_n(self, n, x_partial):
        '''
        n is the number of time steps to propagate forward
        x_partial is the input trajectory [time_dim, bs, state_dim]
        dynamics is a function that takes in x and u and gives the next state
        '''
        h0 = self.initial_rnn_state(x_partial[:1,:,:])

        x_future = []
        u_future = []
        
        o, h = self.lstm(x_partial, h0)    # h is the last hidden state/last output

        x_prev = x_partial[-1:, :,:]    # [1, bs, state_dim]

        for i in range(n):
            u_ = self.proj(h[0])    # [1, bs, ctrl_dim]
            a = (self.a_lim[:,:,1:] - self.a_lim[:,:,:1]) / 2 * u_[:,:,:1] + self.a_lim.mean(-1, keepdims=True)
            delta = (self.delta_lim[:,:,1:] - self.delta_lim[:,:,:1]) / 2 * u_[:,:,1:] + self.delta_lim.mean(-1, keepdims=True)
            u = torch.cat([a, delta], dim=-1)
            u_future.append(u)
            x_next = self.dynamics(x_prev, u, stats=self.stats)    # [1, bs, state_dim]
            x_future.append(x_next)
            o, h = self.lstm(x_next, h)    # o, (h,c) are [1, bs, hidden_dim]

            x_prev = x_next
                
        return torch.cat(x_future, 0), torch.cat(u_future, 0)    # [n, bs, state_dim/ctrl_dim]
        
    
    def state_control_loss(self, x, x_true, u_true, teacher_training=0.0):
        if teacher_training == 0.0:
            o, u, x_pred = self.forward(x)
            return self.L2loss(x_pred, x_true), self.L2loss(u, u_true)
        else:
            prob = np.random.rand(x.shape[0]-1) < teacher_training
            xs = []
            us = []
            xs.append(x[:1,:,:])
            x_input = xs[-1]
            for t in range(x.shape[0]-1):
                o, u, _ = self.forward(x_input)
                x_next = self.dynamics(x_input, u, stats=self.stats)
                xs.append(x_next)
                us.append(u)
                if prob[t]:
                    x_input = x_next
                else:
                    x_input = x[t+1:t+2,:,:]
            xx = torch.cat(xs, 0)
            o, u, _ = self.forward(x_input)
            us.append(u)
            uu = torch.cat(us, 0)
            
            return self.L2loss(xx[1:,:,:], x_true[1:,:,:]), self.L2loss(uu, u_true)
        
    @staticmethod
    def join_partial_future_signal( x_partial, x_future):
        return torch.cat([x_partial, x_future], 0)
    
    def STL_loss_n(self, n, x_partial, formula, formula_input_func, **kwargs):
        '''
        Given partial trajectory, roll out the policy to get a complete trajectory.
        Encourage the complete trajectory to satisfy an stl formula
        '''
        x_future, u_future = self.propagate_n(n, x_partial)    # [n, bs, state_dim/ctrl_dim]
        x_complete = self.join_partial_future_signal(x_partial, x_future)
        signal = unstandardize_data(x_complete, self.stats[0][:,:,:self.state_dim], self.stats[1][:,:,:self.state_dim]).permute([1,0,2]).flip(1)    # [bs, time_dim, state_dim]
        return torch.relu(-formula.robustness(formula_input_func(signal), **kwargs)).mean()
    
    def STL_loss(self, x, formula, formula_input_func, **kwargs):
        signal = unstandardize_data(x, self.stats[0][:,:,:self.state_dim], self.stats[1][:,:,:self.state_dim]).permute([1,0,2]).flip(1)    # [bs, time_dim, state_dim]
        circle =  self.env.obs[0]
        return torch.relu(-formula.robustness(formula_input_func(signal, circle), **kwargs)).mean()
    
    def HJI_loss(self, x_traj):
        '''
        x is [time_dim, bs, state_dim]
        Given a trajectory, compute the integral (finite differencing) of the value function along the trajectory
        Want the value to be negative (more negative the better)
        '''
        
        total_value = self.value_func(unstandardize_data(x_traj, self.stats[0][:,:,:self.state_dim], self.stats[1][:,:,:self.state_dim])).squeeze(-1).relu().sum(0) * self.dt    # [time_dim, bs, 1]

        return total_value.mean()



def outside_circle_stl(signal, circle, device):
    signal = signal.to(device)
    d2 = stlcg.Expression('d2_to_center', (signal[:,:,:2] - torch.tensor(circle.center).unsqueeze(0).unsqueeze(0).to(device)).pow(2).sum(-1, keepdim=True))
    return stlcg.Always(subformula = d2 > circle.radius), d2
    

def in_box_stl(signal, box, device):
    signal = signal.to(device)
    x = stlcg.Expression('x', signal[:,:,:1])
    y = stlcg.Expression('y', signal[:,:,1:2])
    return ((x > box.lower[0]) & (y > box.lower[1])) & ((x < box.upper[0]) & (y < box.upper[1])), ((x, y),(x, y))



def get_formula_input(signal, circle, device):
    signal = signal.to(device)
    d2 = stlcg.Expression('d2_to_center', (signal[:,:,:2] - torch.tensor(circle.center).unsqueeze(0).unsqueeze(0).to(device)).pow(2).sum(-1, keepdim=True))
    x = stlcg.Expression('x', signal[:,:,:1])
    y = stlcg.Expression('y', signal[:,:,1:2])
    return (d2, ((x, y),(x, y)))


def plot_xy_from_tensor(x_train, fig=None):
    xy = x_train.squeeze().detach().numpy()[:,:2]
    if fig is None:
        fig = plt.figure(figsize=(10,8))
    plt.plot(xy[:,0], xy[:,1])
    plt.scatter(xy[:,0], xy[:,1])
    return fig

# if __name__ == '__main__':
  
    # vf = HJIValueFunction.apply
    # inputs = torch.stack([p[:3] for p in points], 1).float().requires_grad_(True)
    # loss = vf(inputs[1:2,:]).squeeze()
    # loss.backward()
    # print(inputs.grad)
    # print(torch.stack([deriv_interp[i]([p[:3] for p in points]) for i in range(4)], 1))


