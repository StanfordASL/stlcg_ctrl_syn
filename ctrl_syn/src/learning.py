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
    μ = torch.tensor(np.mean(data, axis=0, keepdims=True)).float().unsqueeze(1).requires_grad_(False)
    σ = torch.tensor(np.std(data, axis=0, keepdims=True)).float().unsqueeze(1).requires_grad_(False)
    x = torch.tensor(data[:, :4]).float().unsqueeze(1).requires_grad_(False)
    u = torch.tensor(data[:, 4:6]).float().unsqueeze(1).requires_grad_(False)
    return x, u, [μ, σ]


def standardize_data(x, mu, sigma):
    return (x - mu)/sigma

def unstandardize_data(x, mu, sigma):
    return x * sigma + mu

bicycle_params = {"lr" : 0.7, "lf" : 0.5, "V_min" : 0.0, "V_max" : 5.0, "a_min" : -3, "a_max" : 3, "delta_min" : -0.344, "delta_max" : 0.344}

class KinematicBicycle(torch.nn.Module):

    def __init__(self, dt, params={"lr" : 0.7, "lf" : 0.5, "V_min" : 0.0, "V_max" : 5.0, "a_min" : -3, "a_max" : 3, "delta_min" : -0.344, "delta_max" : 0.344}):
        super(KinematicBicycle, self).__init__()
        self.dt = dt
        self.lr = params["lr"]
        self.lf = params["lf"]
        self.V_min = params["V_min"]
        self.V_max = params["V_max"]
        self.a_min = params["a_min"]
        self.a_max = params["a_max"]
        self.delta_min = params["delta_min"]
        self.delta_max = params["delta_max"]
        self.state_dim = 4
        self.ctrl_dim = 2
        
    def forward(self, xcurr, ucurr, tol=1E-3):
        lr = self.lr
        lf = self.lf
        delta_min = self.delta_min
        delta_max = self.delta_max
        V_min = self.V_min
        V_max = self.V_max
        dt = self.dt
        
        x, y, psi, V = x.split(1, dim=-1)
        a, delta = u.split(1, dim=-1)
        a = a.clamp(self.a_min, self.a_max)
        beta = torch.atan(lr / (lr + lf) * torch.tan(delta.clamp(delta_min, delta_max)))
        
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
        return torch.cat([x_new, y_new, psi_new, V_new], dim=-1)




def map_to_interval(x, a, b):
    return x * (b - a) + a

def initial_conditions(n, a, b):
    x0 = torch.rand([n, 1, len(a)])
    return map_to_interval(x0, a, b)


class InitialConditionDataset(torch.utils.data.Dataset):

    def __init__(self, n, a, b):
        self.n = n
        self.ic = initial_conditions(n, a, b)


    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.ic[idx]




def flip_tuple_input(x, time_dim=1):
    if not isinstance(x, tuple):
        return x.flip(time_dim)
    else:
        return (flip_tuple_input(x[0], time_dim), flip_tuple_input(x[1], time_dim))



class STLPolicy(torch.nn.Module):

    def __init__(self, dynamics, hidden_dim, stats, env, dropout=0., num_layers=1, a_min=-3, a_max=3, delta_min=-0.344, delta_max=0.344):
        super(STLPolicy, self).__init__()
        
        self.dynamics = dynamics
        self.stats = stats
        self.dt = dt      
        self.state_dim = dynamics.state_dim
        self.env = env
        self.value_func = value_func
        self.deriv_value_func = deriv_value_func

        a_lim_ = torch.tensor([dynamics.a_min, dynamics.a_max]).float().unsqueeze(0).unsqueeze(0)
        delta_lim_ = torch.tensor([dynamics.delta_min, dynamics.delta_max]).float().unsqueeze(0).unsqueeze(0)
        self.a_lim = standardize_data(a_lim_, stats[0][:,:,4:5], stats[1][:,:,4:5])
        self.delta_lim = standardize_data(delta_lim_, stats[0][:,:,5:], stats[1][:,:,5:])
        
        self.lstm = torch.nn.LSTM(state_dim, hidden_dim, num_layers, dropout=dropout)
        self.proj = torch.nn.Sequential(torch.nn.Linear(hidden_dim, dynamics.ctrl_dim), torch.nn.Tanh())
        self.initialize_rnn_h = torch.nn.Linear(state_dim, hidden_dim)
        self.initialize_rnn_c = torch.nn.Linear(state_dim, hidden_dim)
        self.L2loss = torch.nn.MSELoss()
        ### stl specification details.
        

    def switch_device(self, device):
        self.a_lim = self.a_lim.to(device)
        self.delta_lim = self.delta_lim.to(device)
        self.stats[0] = self.stats[0].to(device)
        self.stats[1] = self.stats[1].to(device)

    def standardize_x(self, x):
        mu =  self.stats[0][:,:,:self.state_dim]
        sigma = self.stats[1][:,:,:self.state_dim]
        return (x - mu)/sigma

    def unstandardize_x(self, x):
        mu =  self.stats[0][:,:,:self.state_dim]
        sigma = self.stats[1][:,:,:self.state_dim]
        return x * sigma + mu

    def standardize_u(self, u):
        mu =  self.stats[0][:,:,self.state_dim:]
        sigma = self.stats[1][:,:,self.state_dim:]
        return (u - mu)/sigma

    def unstandardize_u(self, u):
        mu =  self.stats[0][:,:,self.state_dim:]
        sigma = self.stats[1][:,:,self.state_dim:]
        return u * sigma + mu

    def initial_rnn_state(self, x0):
        # x0 is [bs, state_dim]
        return self.initialize_rnn_h(x0), self.initialize_rnn_c(x0)
    
    def forward(self, x):        
        # x is [time_dim, bs, state_dim]
        h0 = self.initial_rnn_state(x[:1,:,:])

        o, _ = self.lstm(x, h0)    # [time_dim, bs, hidden_dim] , bs = 1 for a single expert trajectory.
        
        # [time_dim, bs, ctrl_dim]  projecting between u_min and u_max (standardize) since proj is between -1 and 1 due to tanh
        u_ = self.proj(o)    # [a, delta]
        # [-1, 1] -> [a, b] : (b - a)/2 * u + (a + b) / 2 
        a = (self.a_lim[:,:,1:] - self.a_lim[:,:,:1]) / 2 * u_[:,:,:1] + self.a_lim.mean(-1, keepdims=True)
        delta = (self.delta_lim[:,:,1:] - self.delta_lim[:,:,:1]) / 2 * u_[:,:,1:] + self.delta_lim.mean(-1, keepdims=True)
        u = torch.cat([a, delta], dim=-1)
        x_next = self.standardize_x(self.dynamics(self.unstandardize_x(x_prev), self.unstandardize_u(u)))    # [1, bs, state_dim]
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
            # [-1, 1] -> [a, b] : (b - a)/2 * u + (a + b) / 2 
            a = (self.a_lim[:,:,1:] - self.a_lim[:,:,:1]) / 2 * u_[:,:,:1] + self.a_lim.mean(-1, keepdims=True)
            delta = (self.delta_lim[:,:,1:] - self.delta_lim[:,:,:1]) / 2 * u_[:,:,1:] + self.delta_lim.mean(-1, keepdims=True)
            u = torch.cat([a, delta], dim=-1)
            u_future.append(u)
            x_next = self.standardize_x(self.dynamics(self.unstandardize_x(x_prev), self.unstandardize_u(u)))    # [1, bs, state_dim]
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
                o, u, x_pred = self.forward(x_input)
                x_next = x_pred[-1:,:,:]
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
    
    def STL_loss_n(self, n, x_partial, formula, formula_input, **kwargs):
        '''
        Given partial trajectory, roll out the policy to get a complete trajectory.
        Encourage the complete trajectory to satisfy an stl formula
        '''
        x_future, u_future = self.propagate_n(n, x_partial)    # [n, bs, state_dim/ctrl_dim]
        x_complete = self.join_partial_future_signal(x_partial, x_future)
        signal = self.unstandardize_x(x_complete).permute([1,0,2]).flip(1)    # [bs, time_dim, state_dim]
        robustness = formula.robustness(formula_input(signal), **kwargs)
        violations = robustness[robustness < 0]
        return -violations.mean()
        # return torch.relu(-formula.robustness(formula_input(signal), **kwargs)).mean()


    def STL_robustness(self, x, formula, formula_input, **kwargs):
        signal = self.unstandardize_x(x).permute([1,0,2])    # [bs, time_dim, state_dim]
        return formula.robustness(formula_input(signal), **kwargs)

    def STL_loss(self, x, formula, formula_input, **kwargs):
        # penalize negative robustness values only
        robustness = self.STL_robustness(x, formula, formula_input, **kwargs)
        violations = robustness[robustness < 0]
        if len(violations) == 0:
            return torch.relu(-robustness).mean()
        return -violations.mean()

    def adversarial_STL_loss(self, x, formula, formula_input, **kwargs):
        # minimize mean robustness
        return self.STL_robustness(x, formula, formula_input, **kwargs).mean()
    


def outside_circle_stl(signal, circle, device):
    signal = signal.to(device)
    d2 = stlcg.Expression('d2_to_center', (signal[:,:,:2] - torch.tensor(circle.center).unsqueeze(0).unsqueeze(0).to(device)).pow(2).sum(-1, keepdim=True))
    return stlcg.Always(subformula = d2 > circle.radius**2), d2

def inside_circle(cover):
    return stlcg.Expression('d2_to_coverage') < cover.radius**2

def always_inside_circle(cover, interval=[0,5]):
    pred = inside_circle(cover)
    return stlcg.Always(subformula=pred, interval=interval)

def inside_circle_input(signal, cover, device='cpu', backwards=False):
    if not backwards:
        signal = signal.flip(1)
    return (signal[:,:,:2].to(device) - torch.tensor(cover.center).to(device)).pow(2).sum(-1, keepdim=True)

def outside_circle(circle):
    return stlcg.Negation(subformula=inside_circle(circle))

def always_outside_circle(circle, interval=None):
    pred = outside_circle(circle)
    return stlcg.Always(subformula=pred, interval=interval)

def outside_circle_input(signal, circle, device='cpu', backwards=False):
    return inside_circle_input(signal, circle, device, backwards)

def get_formula_input(signal, cover, obs, goal, device, backwards=False):
    coverage_input = inside_circle_input(signal, cover, device, backwards)
    obs_input = outside_circle_input(signal, obs, device, backwards)
    goal_input = inside_circle_input(signal, goal, device, backwards)
    speed_input = signal[:,:,3:4]
    if not backwards:
        speed_input = speed_input.flip(1)
    return (((coverage_input, speed_input),(goal_input, speed_input)), obs_input)

def in_box_stl(signal, box, device):
    signal = signal.to(device)
    x = stlcg.Expression('x', signal[:,:,:1])
    y = stlcg.Expression('y', signal[:,:,1:2])
    return ((x > box.lower[0]) & (y > box.lower[1])) & ((x < box.upper[0]) & (y < box.upper[1])), ((x, y),(x, y))


def stop_in_box_stl(signal, box, device):
    signal = signal.to(device)
    x = stlcg.Expression('x', signal[:,:,:1])
    y = stlcg.Expression('y', signal[:,:,1:2])
    v = stlcg.Expression('v', signal[:,:,-1:])
    return (((x > box.lower[0]) & (y > box.lower[1])) & ((x < box.upper[0]) & (y < box.upper[1]))) & ((v >= 0.0) & (v < 0.5)), (((x, y),(x, y)), (v,v))



def get_goal_formula_input(signal, circle, device):
    signal = signal.to(device)
    d2 = stlcg.Expression('d2_to_center', (signal[:,:,:2] - torch.tensor(circle.center).unsqueeze(0).unsqueeze(0).to(device)).pow(2).sum(-1, keepdim=True))
    x = stlcg.Expression('x', signal[:,:,:1])
    y = stlcg.Expression('y', signal[:,:,1:2])
    return (d2, ((x, y),(x, y)))

def get_coverage_formula_input(signal, circle, device):
    d2, xy = get_goal_formula_input(signal, circle, device)
    return (((xy, xy), xy), d2)

def get_test_formula_input(signal, circle, device):
    signal = signal.to(device)
    x = stlcg.Expression('x', signal[:,:,:1])
    y = stlcg.Expression('y', signal[:,:,1:2])
    v = stlcg.Expression('v', signal[:,:,-1:])
    return (((x, y),(x, y)), (v,v))

def get_coverage_test_formula_input(signal, circle, device):
    # (cov & end) & obs
    d2, xy = get_goal_formula_input(signal, circle, device)
    stop_in_end_goal_input = get_test_formula_input(signal, circle, device)
    return ((xy, stop_in_end_goal_input), d2)


