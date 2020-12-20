import sys
sys.path.append('../../../stlcg_karen/src')
sys.path.append('../../expert_demo_ros/src/utils')
sys.path.append('../')
import stlcg
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision

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

def prepare_data(npy_file, batch_dim=0):
    '''
    data is a numpy of size [time_dim, 7]. The columns are: [t, x, y, psi, V, a, delta]
    This extracts the states and controls and turns them into tensors of size [time_dim, 1, state/ctrl_dim]
    This also outputs the mean and std of the data [1, 1, state+ctrl_dim]
    '''
    data = np.load(npy_file)[:,1:]
    μ = torch.tensor(np.mean(data, axis=0, keepdims=True)).float().unsqueeze(batch_dim).requires_grad_(False)
    σ = torch.tensor(np.std(data, axis=0, keepdims=True)).float().unsqueeze(batch_dim).requires_grad_(False)
    x = torch.tensor(data[:, :4]).float().unsqueeze(batch_dim).requires_grad_(False)
    u = torch.tensor(data[:, 4:6]).float().unsqueeze(batch_dim).requires_grad_(False)
    return x, u, [μ, σ]

def convert_env_img_to_tensor(img_path, grey=torchvision.transforms.Grayscale(), resize=torchvision.transforms.Resize([480, 480])):
    image_tensor = torch.stack([TF.to_tensor(resize(grey(Image.open(img_path + '/init.png')))),
                         TF.to_tensor(resize(grey(Image.open(img_path + '/final.png')))),
                         TF.to_tensor(resize(grey(Image.open(img_path + '/covers.png')))),
                         TF.to_tensor(resize(grey(Image.open(img_path + '/obs.png'))))], dim=1)

    return image_tensor

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

        x, y, psi, V = xcurr.split(1, dim=-1)
        a, delta = ucurr.split(1, dim=-1)
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

    def __init__(self, dynamics, hidden_dim, stats, env, dropout=0., num_layers=1):
        super(STLPolicy, self).__init__()

        self.dynamics = dynamics
        self.stats = stats
        self.dt = dynamics.dt
        self.state_dim = dynamics.state_dim
        self.env = env

        a_lim_ = torch.tensor([dynamics.a_min, dynamics.a_max]).float().unsqueeze(0).unsqueeze(0)
        delta_lim_ = torch.tensor([dynamics.delta_min, dynamics.delta_max]).float().unsqueeze(0).unsqueeze(0)
        self.a_lim = standardize_data(a_lim_, stats[0][:,:,4:5], stats[1][:,:,4:5])
        self.delta_lim = standardize_data(delta_lim_, stats[0][:,:,5:], stats[1][:,:,5:])

        self.lstm = torch.nn.LSTM(self.state_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.proj = torch.nn.Sequential(torch.nn.Linear(hidden_dim, dynamics.ctrl_dim), torch.nn.Tanh())
        self.initialize_rnn_h = torch.nn.Linear(self.state_dim, hidden_dim)
        self.initialize_rnn_c = torch.nn.Linear(self.state_dim, hidden_dim)
        self.L2loss = torch.nn.MSELoss()

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

    def forward(self, x0, h0=None):
        '''
        Passes x through the LSTM, computes control u, and propagate through dynamics. Very vanilla propagation. Feeds in x into LSTM.

        Inputs:
            x is [bs, time_dim, state_dim]

        Outputs:
            o is [bs, time_dim, hidden_dim] --- outputs of the LSTM cell
            u is [bs, time_dim, ctrl_dim] --- projection of LSTM output state to control
            x_next is [bs, time_dim, state_dim] --- propagate dynamics from previous state and computed controls
        '''
        if h0 is None:
            h0 = self.initial_rnn_state(x0[:,:1,:].permute([1,0,2]))

        o, _ = self.lstm(x0, h0)    # [bs, time_dim, hidden_dim] , bs = 1 for a single expert trajectory.

        # [bs, time_dim, ctrl_dim]  projecting between u_min and u_max (standardize) since proj is between -1 and 1 due to tanh
        u_ = self.proj(o)    # [bs, 1, ctrl_dim]   [a, delta]

        # [-1, 1] -> [a, b] : (b - a)/2 * u + (a + b) / 2
        a_min, a_max = self.a_lim.split(1, dim=-1)
        delta_min, delta_max = self.delta_lim.split(1, dim=-1)
        a = (a_max - a_min) / 2 * u_[:,:,:1] + self.a_lim.mean(-1, keepdims=True)
        delta = (delta_max - delta_min) / 2 * u_[:,:,1:] + self.delta_lim.mean(-1, keepdims=True)
        u = torch.cat([a, delta], dim=-1)
        # propagate dynamics
        x_next = self.standardize_x(self.dynamics(self.unstandardize_x(x0), self.unstandardize_u(u)))    # [1, bs, state_dim]
        # append outputs to initial state
        # x_pred = self.join_partial_future_signal(x0[:,:1,:], x_next[:,:-1,:])
        return o, u, x_next



    def propagate_n(self, n, x_partial, time_dim=1):
        '''
        Given x_partial, predict the future controls/states for the next n time steps

        Inputs:
            n is the number of time steps to propagate forward
            x_partial is [bs, previous_time_dim, state_dim] --- input partial trajectory
            time_dim --- dimension of time. Default=1

        Outputs:
            x_next is [bs, n, state_dim] --- sequence of states over the next n time steps
            u_next is [bs, n, ctrl_dim] --- sequence of controls over the next n time steps
        '''

        h0 = self.initial_rnn_state(x_partial[:,:1,:].permute([1,0,2]))

        x_future = []
        u_future = []

        o, h = self.lstm(x_partial, h0)    # h is the last hidden state/last output
        # get last state, as that is the input to compute the first step of the n steps
        x_prev = x_partial[:,-1:,:]    # [bs, 1, state_dim]

        for i in range(n):
            u_ = self.proj(o)    # [bs, 1, ctrl_dim]
            # [-1, 1] -> [a, b] : (b - a)/2 * u + (a + b) / 2
            a_min, a_max = self.a_lim.split(1, dim=-1)
            delta_min, delta_max = self.delta_lim.split(1, dim=-1)
            a = (a_max - a_min) / 2 * u_[:,:,:1] + self.a_lim.mean(-1, keepdims=True)
            delta = (delta_max - delta_min) / 2 * u_[:,:,1:] + self.delta_lim.mean(-1, keepdims=True)
            u = torch.cat([a, delta], dim=-1)
            u_future.append(u)
            x_next = self.standardize_x(self.dynamics(self.unstandardize_x(x_prev), self.unstandardize_u(u)))    # [1, bs, state_dim]
            x_future.append(x_next)
            o, h = self.lstm(x_next, h)    # o, (h,c) are [1, bs, hidden_dim]

            x_prev = x_next
        x_next = torch.cat(x_future, time_dim)
        u_next = torch.cat(u_future, time_dim)
        return x_next, u_next


    def reconstruction_loss(self, x_true, u_true, teacher_training=0.0, time_dim=1):
        '''
        Given an input trajectory x_traj, compute the reconstruction error for state and control.

        Inputs:
            x_true is [bs, time_dim, state_dim] --- input state trajectory (from expert demonstration)
            u_true is [bs, time_dim, ctrl_dim] --- input control trajectory (from expert demonstration)
            teacher_training ∈ [0,1] --- a probability of using the previous propagated state as opposed to the true state from x_true
                                         a run time, teacher_training=1.0 since we do not have access to the ground truth
            time_dim --- dimension of time. Default=1

        Outputs:
            recon_state_loss is the MSE reconstruction loss over the states
            recon_ctrl_loss is the MSE reconstruction loss over the controls
            xx is [bs, time_dim, state_dim] --- predicted sequence of states, aimed at recovering x_true
            uu is [bs, time_dim, ctrl_dim] --- predicted sequence of controls, aimed at recovering u_true
        '''
        # no teacher training, relying on ground truth
        if teacher_training == 0.0:
            o, u_pred, x_pred = self.forward(x_true)
            x_pred = self.join_partial_future_signal(x_true[:,:1,:], x_pred[:,:-1,:])
            return self.L2loss(x_pred, x_true), self.L2loss(u_pred, u_true), x_pred, u_pred
        else:
            prob = np.random.rand(x_true.shape[1]-1) < teacher_training
            xs = []
            us = []
            xs.append(x_true[:,:1,:])
            x_input = xs[-1]
            for t in range(x_true.shape[1]-1):
                o, u, x_next = self.forward(x_input)
                xs.append(x_next)
                us.append(u)
                if prob[t]:
                    x_input = x_next
                else:
                    x_input = x_true[:,t+1:t+2,:]
            xx = torch.cat(xs, time_dim)
            o, u, _ = self.forward(x_input)
            us.append(u)
            uu = torch.cat(us, time_dim)
            # ignoring the first time step since there will be no error
            recon_state_loss = self.L2loss(xx[:,1:,:], x_true[:,1:,:])
            recon_ctrl_loss = self.L2loss(uu, u_true)
            return recon_state_loss, recon_ctrl_loss, xx, uu


    @staticmethod
    def join_partial_future_signal( x_partial, x_future, time_dim=1):
        return torch.cat([x_partial, x_future], time_dim)

    def STL_robustness(self, x, formula, formula_input, **kwargs):
        signal = self.unstandardize_x(x)    # [bs, time_dim, state_dim]
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


class STLCNNPolicy(torch.nn.Module):

    def __init__(self, dynamics, hidden_dim, stats, env, dropout=0., num_layers=1):
        super(STLPolicy, self).__init__()

        self.dynamics = dynamics
        self.stats = stats
        self.dt = dynamics.dt
        self.state_dim = dynamics.state_dim
        self.env = env

        a_lim_ = torch.tensor([dynamics.a_min, dynamics.a_max]).float().unsqueeze(0).unsqueeze(0)
        delta_lim_ = torch.tensor([dynamics.delta_min, dynamics.delta_max]).float().unsqueeze(0).unsqueeze(0)
        self.a_lim = standardize_data(a_lim_, stats[0][:,:,4:5], stats[1][:,:,4:5])
        self.delta_lim = standardize_data(delta_lim_, stats[0][:,:,5:], stats[1][:,:,5:])
        
        self.cnn = torch.nn.Sequential(torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=8, padding=4, stride=4),
                                       torch.nn.BatchNorm2d(4),
                                       torch.nn.ReLU(),
                                       torch.nn.MaxPool2d(kernel_size=4),
                                       torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=8, padding=4, stride=4))


        self.lstm = torch.nn.LSTM(self.state_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.proj = torch.nn.Sequential(torch.nn.Linear(hidden_dim, dynamics.ctrl_dim), torch.nn.Tanh())
        self.initialize_rnn_h = torch.nn.Sequential(torch.nn.Linear(8 * 8, hidden_dim),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(hidden_dim, hidden_dim),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(hidden_dim, hidden_dim))
        self.initialize_rnn_c = torch.nn.Sequential(torch.nn.Linear(8 * 8, hidden_dim),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(hidden_dim, hidden_dim),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(hidden_dim, hidden_dim))
        self.L2loss = torch.nn.MSELoss()

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

    def forward(self, x0, h0=None):
        '''
        Passes x through the LSTM, computes control u, and propagate through dynamics. Very vanilla propagation. Feeds in x into LSTM.

        Inputs:
            x is [bs, time_dim, state_dim]

        Outputs:
            o is [bs, time_dim, hidden_dim] --- outputs of the LSTM cell
            u is [bs, time_dim, ctrl_dim] --- projection of LSTM output state to control
            x_next is [bs, time_dim, state_dim] --- propagate dynamics from previous state and computed controls
        '''
        if h0 is None:
            h0 = self.initial_rnn_state(x0[:,:1,:].permute([1,0,2]))

        o, _ = self.lstm(x0, h0)    # [bs, time_dim, hidden_dim] , bs = 1 for a single expert trajectory.

        # [bs, time_dim, ctrl_dim]  projecting between u_min and u_max (standardize) since proj is between -1 and 1 due to tanh
        u_ = self.proj(o)    # [bs, 1, ctrl_dim]   [a, delta]

        # [-1, 1] -> [a, b] : (b - a)/2 * u + (a + b) / 2
        a_min, a_max = self.a_lim.split(1, dim=-1)
        delta_min, delta_max = self.delta_lim.split(1, dim=-1)
        a = (a_max - a_min) / 2 * u_[:,:,:1] + self.a_lim.mean(-1, keepdims=True)
        delta = (delta_max - delta_min) / 2 * u_[:,:,1:] + self.delta_lim.mean(-1, keepdims=True)
        u = torch.cat([a, delta], dim=-1)
        # propagate dynamics
        x_next = self.standardize_x(self.dynamics(self.unstandardize_x(x0), self.unstandardize_u(u)))    # [1, bs, state_dim]
        # append outputs to initial state
        # x_pred = self.join_partial_future_signal(x0[:,:1,:], x_next[:,:-1,:])
        return o, u, x_next



    def propagate_n(self, n, x_partial, time_dim=1):
        '''
        Given x_partial, predict the future controls/states for the next n time steps

        Inputs:
            n is the number of time steps to propagate forward
            x_partial is [bs, previous_time_dim, state_dim] --- input partial trajectory
            time_dim --- dimension of time. Default=1

        Outputs:
            x_next is [bs, n, state_dim] --- sequence of states over the next n time steps
            u_next is [bs, n, ctrl_dim] --- sequence of controls over the next n time steps
        '''

        h0 = self.initial_rnn_state(x_partial[:,:1,:].permute([1,0,2]))

        x_future = []
        u_future = []

        o, h = self.lstm(x_partial, h0)    # h is the last hidden state/last output
        # get last state, as that is the input to compute the first step of the n steps
        x_prev = x_partial[:,-1:,:]    # [bs, 1, state_dim]

        for i in range(n):
            u_ = self.proj(o)    # [bs, 1, ctrl_dim]
            # [-1, 1] -> [a, b] : (b - a)/2 * u + (a + b) / 2
            a_min, a_max = self.a_lim.split(1, dim=-1)
            delta_min, delta_max = self.delta_lim.split(1, dim=-1)
            a = (a_max - a_min) / 2 * u_[:,:,:1] + self.a_lim.mean(-1, keepdims=True)
            delta = (delta_max - delta_min) / 2 * u_[:,:,1:] + self.delta_lim.mean(-1, keepdims=True)
            u = torch.cat([a, delta], dim=-1)
            u_future.append(u)
            x_next = self.standardize_x(self.dynamics(self.unstandardize_x(x_prev), self.unstandardize_u(u)))    # [1, bs, state_dim]
            x_future.append(x_next)
            o, h = self.lstm(x_next, h)    # o, (h,c) are [1, bs, hidden_dim]

            x_prev = x_next
        x_next = torch.cat(x_future, time_dim)
        u_next = torch.cat(u_future, time_dim)
        return x_next, u_next


    def reconstruction_loss(self, x_true, u_true, teacher_training=0.0, time_dim=1):
        '''
        Given an input trajectory x_traj, compute the reconstruction error for state and control.

        Inputs:
            x_true is [bs, time_dim, state_dim] --- input state trajectory (from expert demonstration)
            u_true is [bs, time_dim, ctrl_dim] --- input control trajectory (from expert demonstration)
            teacher_training ∈ [0,1] --- a probability of using the previous propagated state as opposed to the true state from x_true
                                         a run time, teacher_training=1.0 since we do not have access to the ground truth
            time_dim --- dimension of time. Default=1

        Outputs:
            recon_state_loss is the MSE reconstruction loss over the states
            recon_ctrl_loss is the MSE reconstruction loss over the controls
            xx is [bs, time_dim, state_dim] --- predicted sequence of states, aimed at recovering x_true
            uu is [bs, time_dim, ctrl_dim] --- predicted sequence of controls, aimed at recovering u_true
        '''
        # no teacher training, relying on ground truth
        if teacher_training == 0.0:
            o, u_pred, x_pred = self.forward(x_true)
            x_pred = self.join_partial_future_signal(x_true[:,:1,:], x_pred[:,:-1,:])
            return self.L2loss(x_pred, x_true), self.L2loss(u_pred, u_true), x_pred, u_pred
        else:
            prob = np.random.rand(x_true.shape[1]-1) < teacher_training
            xs = []
            us = []
            xs.append(x_true[:,:1,:])
            x_input = xs[-1]
            for t in range(x_true.shape[1]-1):
                o, u, x_next = self.forward(x_input)
                xs.append(x_next)
                us.append(u)
                if prob[t]:
                    x_input = x_next
                else:
                    x_input = x_true[:,t+1:t+2,:]
            xx = torch.cat(xs, time_dim)
            o, u, _ = self.forward(x_input)
            us.append(u)
            uu = torch.cat(us, time_dim)
            # ignoring the first time step since there will be no error
            recon_state_loss = self.L2loss(xx[:,1:,:], x_true[:,1:,:])
            recon_ctrl_loss = self.L2loss(uu, u_true)
            return recon_state_loss, recon_ctrl_loss, xx, uu


    @staticmethod
    def join_partial_future_signal( x_partial, x_future, time_dim=1):
        return torch.cat([x_partial, x_future], time_dim)

    def STL_robustness(self, x, formula, formula_input, **kwargs):
        signal = self.unstandardize_x(x)    # [bs, time_dim, state_dim]
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
