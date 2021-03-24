import sys
sys.path.append('../../../stlcg_karen/src')
sys.path.append('../../expert_demo_ros/src/utils')
sys.path.append('../')
import stlcg
import matplotlib.pyplot as plt

import os

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms.functional as TF
import torchvision

import numpy as np
import torch
import scipy.io as sio

# from torch_interpolations.torch_interpolations.multilinear import RegularGridInterpolator
from torch.utils.tensorboard import SummaryWriter

from environment import *
import IPython
import glob

pardir = os.path.dirname(os.path.dirname(__file__))



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
    mu = torch.tensor(np.mean(data, axis=0, keepdims=True)).float().unsqueeze(batch_dim).requires_grad_(False)
    sigma = torch.tensor(np.std(data, axis=0, keepdims=True)).float().unsqueeze(batch_dim).requires_grad_(False)
    x = torch.tensor(data[:, :4]).float().unsqueeze(batch_dim).requires_grad_(False)
    u = torch.tensor(data[:, 4:6]).float().unsqueeze(batch_dim).requires_grad_(False)
    return x, u, [mu, sigma]


def prepare_data_img(case, filedir, batch_dim=0, height=480, width=480, dpi=500):
    if case == "coverage":
        data_tls = [np.load(fi).shape[0] for fi in sorted(glob.glob(filedir+'*'))]
        max_tl = max(data_tls)
        n_data = len(data_tls)
        data = torch.zeros([n_data, max_tl, 6]).requires_grad_(False)
        data_list = []
        imgs = torch.zeros([n_data, 4, height, width]).requires_grad_(False)
        tls = torch.zeros([n_data]).requires_grad_(False)
        centers = torch.zeros([n_data]).requires_grad_(False)

        for (j,fi) in enumerate(sorted(glob.glob(filedir+'*'))):
            fi_split = fi.split('_')
            i = fi_split[-2]

            imgs[j,:,:,:] = convert_env_img_to_tensor(os.path.join(pardir, "figs/environments/%.1f"%float(i)))
            data_j = torch.tensor(np.load(fi)[:,1:]).float()
            data_list.append(data_j)
            traj_length = data_j.shape[0]
            data[j,:traj_length, :] = data_j

            tls[j] = traj_length
            centers[j] = float(i)

        mu = torch.cat(data_list, dim=0).mean(0).unsqueeze(0).unsqueeze(0)
        sigma = torch.cat(data_list, dim=0).std(0).unsqueeze(0).unsqueeze(0)

    elif case == "drive":
        data_tls = [np.load(fi).shape[0] for fi in sorted(glob.glob(filedir+'*'))]
        max_tl = max(data_tls)
        n_data = len(data_tls)
        data = torch.zeros([n_data, max_tl, 6]).requires_grad_(False)
        data_list = []
        imgs = torch.zeros([n_data, 4, height, width]).requires_grad_(False)
        tls = torch.zeros([n_data]).requires_grad_(False)
        centers = torch.zeros([n_data, 2]).requires_grad_(False)
        ps = [[8, -10], [-10, 6], [8, 4], [7, -10], [-10, 7], [3, 7], [3, -10], [10, -10]]

        for (j,fi) in enumerate(sorted(glob.glob(filedir+'*'))):
            fi_split = fi.split('_')
            i = fi_split[-1][0]
            imgs[j,:,:,:] = generate_img_tensor_from_parameter(case, ps[j], width=width, height=height, dpi=dpi)
            data_j = torch.tensor(np.load(fi)[:,1:]).float()
            data_list.append(data_j)
            traj_length = data_j.shape[0]
            data[j,:traj_length, :] = data_j

            tls[j] = traj_length
            centers[j] = torch.tensor(ps[j]).float()

        mu = torch.cat(data_list, dim=0).mean(0).unsqueeze(0).unsqueeze(0)
        sigma = torch.cat(data_list, dim=0).std(0).unsqueeze(0).unsqueeze(0)
    return data[:,:,:4], data[:,:,4:6], tls, imgs, [mu, sigma], centers

def convert_env_img_to_tensor(img_path, grey=torchvision.transforms.Grayscale(), resize=torchvision.transforms.Resize([480, 480])):
    image_tensor = torch.stack([TF.to_tensor(resize(grey(Image.open(img_path + '/init.png')))),
                         TF.to_tensor(resize(grey(Image.open(img_path + '/final.png')))),
                         TF.to_tensor(resize(grey(Image.open(img_path + '/covers.png')))),
                         TF.to_tensor(resize(grey(Image.open(img_path + '/obs.png'))))], dim=1)

    return image_tensor



def generate_img_tensor(env, width=480, height=480, dpi=500, xlim=[-5,15], ylim=[-5,15]):

    plt_params = {"color": "black", "fill": True}
    figsize = (width/dpi, height/dpi)

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    fig.subplots_adjust(0,0,1,1)
    _, ax = env.initial.draw2D(ax=ax, **plt_params)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    canvas.draw()
    X = np.array(fig.canvas.renderer._renderer)[:,:,:1]/255

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    fig.subplots_adjust(0,0,1,1)
    _, ax = env.final.draw2D( ax=ax, **plt_params)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    canvas.draw()
    X = np.concatenate([X, np.array(fig.canvas.renderer._renderer)[:,:,:1]/255], axis=-1)

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    fig.subplots_adjust(0,0,1,1)
    for covs in env.covers:
        _, ax = covs.draw2D( ax=ax, **plt_params)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    canvas.draw()
    X = np.concatenate([X, np.array(fig.canvas.renderer._renderer)[:,:,:1]/255], axis=-1)

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    fig.subplots_adjust(0,0,1,1)
    for obs in env.obs:
        _, ax = obs.draw2D( ax=ax, **plt_params)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    canvas.draw()
    X = np.concatenate([X, np.array(fig.canvas.renderer._renderer)[:,:,:1]/255], axis=-1)
    
    return torch.tensor(X).permute(2,0,1).float()



def sample_environment_parameters(case, n):
    if case == "coverage":
        return np.round(1+np.random.rand(n) * 9, 2)    # between 1-10, and two decimal place
    elif case == "drive":
        return np.concatenate([generate_random_drive_parameters() for i in range(n)], axis=0)
    else:
        raise Exception("Case %s does not exist"%case)



def update_environment(case, env, p, ic_bs, carlength=1.2):
    if case == "coverage":
        centers = p
        if isinstance(centers, np.ndarray):
            final_x = env.final.center[0]
            obs_x = (centers + final_x) / 2
            env.covers[0].center = torch.tensor(np.stack([centers, 3.5 * np.ones_like(centers)], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])
            env.obs[0].center = torch.tensor(np.stack([obs_x, 9. * np.ones_like(obs_x)], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])
        else:
            final_x = env.final.center[0]
            obs_x = (centers + final_x) / 2
            env.covers[0].center = [centers, 3.5 ]
            env.obs[0].center = [obs_x, 9. ]

        return centers
    elif case == "drive":
        obs = p
        if len(obs.shape) > 1:
            # env.obs[2].lower = torch.tensor(np.stack([np.ones_like(obs[:,0]) * 5.2, obs[:,0]], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])
            # env.obs[2].upper = torch.tensor(np.stack([np.ones_like(obs[:,0]) * 5.8, (obs[:,0] + carlength)], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])

            # env.obs[3].lower = torch.tensor(np.stack([np.ones_like(obs[:,1]) * 6.2, obs[:,1]], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])
            # env.obs[3].upper = torch.tensor(np.stack([np.ones_like(obs[:,1]) * 6.8, (obs[:,1] + carlength)], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])

            env.obs[2].center = torch.tensor(np.stack([np.ones_like(obs[:,0]) * 5.5, obs[:,0]], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])
            env.obs[3].center = torch.tensor(np.stack([np.ones_like(obs[:,1]) * 6.5, obs[:,1]], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])
        else:
            # env.obs[2].lower = [5.2, obs[0]]
            # env.obs[2].upper = [5.8, obs[0] + carlength]

            # env.obs[3].lower = [6.2, obs[1]]
            # env.obs[3].upper = [6.8, obs[1] + carlength]
            env.obs[2].center = [5.5, obs[0]]
            env.obs[3].center = [6.5, obs[1]]
        return obs
    else:
        raise Exception("Case %s does not exist"%case)


def append_environment(case, env, p, ic_bs, carlength=1.2):
    if case == "coverage":
        centers = p
        final_x = env.final.center[0]
        obs_x = (centers + final_x) / 2
        env.covers[0].center = torch.cat([env.covers[0].center, torch.tensor(np.stack([centers, 3.5 * np.ones_like(centers)], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])], dim=0)
        env.obs[0].center = torch.cat([env.obs[0].center, torch.tensor(np.stack([obs_x, 9. * np.ones_like(obs_x)], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])], dim=0)
        return centers
    elif case == "drive":
        obs = p
        # env.obs[2].lower = torch.cat([env.obs[2].lower, torch.tensor(np.stack([np.ones_like(obs[:,0]) * 5.2, obs[:,0]], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])], dim=0)
        # env.obs[2].upper = torch.cat([env.obs[2].upper, torch.tensor(np.stack([np.ones_like(obs[:,0]) * 5.8, (obs[:,0] + carlength)], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])], dim=0)

        # env.obs[3].lower = torch.cat([env.obs[3].lower, torch.tensor(np.stack([np.ones_like(obs[:,1]) * 6.2, obs[:,1]], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])], dim=0)
        # env.obs[3].upper = torch.cat([env.obs[3].upper, torch.tensor(np.stack([np.ones_like(obs[:,1]) * 6.8, (obs[:,1] + carlength)], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])], dim=0)

        env.obs[2].center = torch.cat([env.obs[2].center, torch.tensor(np.stack([np.ones_like(obs[:,0]) * 5.5, obs[:,0]], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])], dim=0)
        env.obs[3].center = torch.cat([env.obs[3].center, torch.tensor(np.stack([np.ones_like(obs[:,1]) * 6.5, obs[:,1]], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, ic_bs, 1, 1]).view([-1, 1, 2])], dim=0)
        return obs
    else:
        raise Exception("Case %s does not exist"%case)



def sample_and_update_environment(case, env, img_bs, ic_bs, carlength=1.2):
    p = sample_environment_parameters(case, img_bs)
    return update_environment(case, env, p, ic_bs, carlength=carlength)


def generate_img_tensor_from_parameter(case, p, carlength=1.2, width=480, height=480, dpi=500):
    env = generate_env_from_parameters(case, p, carlength=carlength)
    if case == "coverage":
        xlim = [-5, 15]
        ylim = [-5, 15]
    elif case == "drive":
        xlim = [-2, 14]
        ylim = [0, 16]
    return generate_img_tensor(env, width=width, height=height, dpi=dpi, xlim=xlim, ylim=ylim)


def generate_random_drive_parameters(carlength=1.2):
    left1 = np.random.rand() * 6 + 2
    right2 = np.random.rand() * 5 + 3
    right3 = np.random.rand(1) * 3 + 5
    while True:
        left3 = np.random.rand(1) * 10 + 1
        if np.abs(right3 - left3) > (carlength + 2):
            break
    return np.array([[left1, -10],[-10, right2],[left3.item(), right3.item()]])


def generate_env_from_parameters(case, p, carlength=1.2):
    if case == "coverage":
        final_x = 5.0
        obs_x = (p + final_x) / 2
        params = { "covers": [Circle([p, 3.5], 2.0)],
           "obstacles": [Circle([obs_x, 9.], 1.5)],
           "initial": Box([2, -4.],[8, -2]),
           "final": Circle([final_x, 13], 1.0)
        } 
        return Environment(params)

    elif case == "drive":
        # car = [Box([5.2, p[0]], [5.8, p[0] + carlength]), Box([6.2, p[1]], [6.8, p[1] + carlength])]
        car = [Circle([5.5, p[0]], 0.5), Circle([6.5, p[1]], 0.5)]
            
        params = { "covers": [],
                   "obstacles": [Box([-2,0], [5, 16]), Box([7,0], [14, 16])] + car,
                   "initial": Box([6.35, 0.0],[6.65, 1.0]),
                   "final": Box([6.35, 11.0],[6.65, 16.0])
                } 
        return Environment(params)

def standardize_data(x, mu, sigma):
    return (x - mu)/sigma

def unstandardize_data(x, mu, sigma):
    return x * sigma + mu

bicycle_params = {"lr" : 0.7, "lf" : 0.5, "V_min" : 0.0, "V_max" : 5.0, "a_min" : -3, "a_max" : 3, "delta_min" : -0.344, "delta_max" : 0.344}

class KinematicBicycle(torch.nn.Module):

    def __init__(self, dt, params={"lr" : 0.7, "lf" : 0.5, "V_min" : 0.0, "V_max" : 5.0, "a_min" : -3, "a_max" : 3, "delta_min" : -0.344, "delta_max" : 0.344, "disturbance_scale" : [0.0, 0.0]}):
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
        self.disturbance_dist = torch.distributions.Normal(torch.tensor([0.0, 0.0]), torch.tensor(params["disturbance_scale"]))

    def forward(self, xcurr, ucurr, tol=1E-3):
        lr = self.lr
        lf = self.lf
        delta_min = self.delta_min
        delta_max = self.delta_max
        V_min = self.V_min
        V_max = self.V_max
        dt = self.dt
        dcurr = self.disturbance_dist.sample(xcurr.shape[:-1]).to(xcurr.device).float()
        da, ddelta = dcurr.split(1, dim=-1)
        x, y, psi, V = xcurr.split(1, dim=-1)
        a, delta = ucurr.split(1, dim=-1)
        a = (a + da).clamp(self.a_min, self.a_max)
        delta = (delta + ddelta).clamp(delta_min, delta_max)
        beta = torch.atan(lr / (lr + lf) * torch.tan(delta))

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


class InitialConditionDatasetCNN(torch.utils.data.Dataset):

    def __init__(self, initial_conditions, adv_ic=None, adv_img_p=None):
        self.ic = initial_conditions
        self.adv_ic = adv_ic
        self.adv_img_p = adv_img_p


    def __len__(self):
        return self.ic.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.adv_ic is None:
            return {"ic": self.ic[idx]}
        else:
            return {"ic": self.ic[idx], "adv_ic": self.adv_ic[idx], "adv_img_p": self.adv_img_p[idx]}

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
        # note the dimensions!
        # x0 is [1, bs, state_dim]
        return self.initialize_rnn_h(x0), self.initialize_rnn_c(x0)

    def forward(self, x0, h0=None):
        '''
        Passes x through the LSTM, computes control u, and propagate through dynamics. Very vanilla propagation. Feeds in x into LSTM.

        Inputs:
            x0 is [bs, time_dim, state_dim]

        Outputs:
            o is [bs, time_dim, hidden_dim] --- outputs of the LSTM cell
            u is [bs, time_dim, ctrl_dim] --- projection of LSTM output state to control
            x_next is [bs, time_dim, state_dim] --- propagate dynamics from previous state and computed controls
        '''
        if h0 is None:
            h0 = self.initial_rnn_state(x0[:,:1,:].permute([1,0,2]))

        o, h0 = self.lstm(x0, h0)    # [bs, time_dim, hidden_dim] , bs = 1 for a single expert trajectory.

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
        return o, u, x_next, h0



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
            teacher_training in [0,1] --- a probability of using the previous propagated state as opposed to the true state from x_true
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
            h0 = None
            for t in range(x_true.shape[1]-1):
                o, u, x_next, h0 = self.forward(x_input, h0=h0)
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

def get_tl_elements(data, tls):
    arange = torch.ones_like(data) * torch.arange(tls.max()).to(data.device).unsqueeze(0).unsqueeze(-1)
    mask = (arange == (tls - 1).unsqueeze(-1).unsqueeze(-1))
    return torch.masked_select(data, mask).view(data.shape[0], 1, data.shape[2])





                                                                     
                                                                     
#         CCCCCCCCCCCCCNNNNNNNN        NNNNNNNNNNNNNNNN        NNNNNNNN
#      CCC::::::::::::CN:::::::N       N::::::NN:::::::N       N::::::N
#    CC:::::::::::::::CN::::::::N      N::::::NN::::::::N      N::::::N
#   C:::::CCCCCCCC::::CN:::::::::N     N::::::NN:::::::::N     N::::::N
#  C:::::C       CCCCCCN::::::::::N    N::::::NN::::::::::N    N::::::N
# C:::::C              N:::::::::::N   N::::::NN:::::::::::N   N::::::N
# C:::::C              N:::::::N::::N  N::::::NN:::::::N::::N  N::::::N
# C:::::C              N::::::N N::::N N::::::NN::::::N N::::N N::::::N
# C:::::C              N::::::N  N::::N:::::::NN::::::N  N::::N:::::::N
# C:::::C              N::::::N   N:::::::::::NN::::::N   N:::::::::::N
# C:::::C              N::::::N    N::::::::::NN::::::N    N::::::::::N
#  C:::::C       CCCCCCN::::::N     N:::::::::NN::::::N     N:::::::::N
#   C:::::CCCCCCCC::::CN::::::N      N::::::::NN::::::N      N::::::::N
#    CC:::::::::::::::CN::::::N       N:::::::NN::::::N       N:::::::N
#      CCC::::::::::::CN::::::N        N::::::NN::::::N        N::::::N
#         CCCCCCCCCCCCCNNNNNNNN         NNNNNNNNNNNNNNN         NNNNNNN
                                                                     
                                                                     
                                                                     
                                                                     
        




class STLCNNPolicy(torch.nn.Module):

    def __init__(self, dynamics, hidden_dim, stats, env, dropout=0., num_layers=1):
        super(STLCNNPolicy, self).__init__()

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
                                       torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=8, padding=4, stride=4))


        self.lstm = torch.nn.LSTM(self.state_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        # self.proj = torch.nn.Sequential(torch.nn.Linear(hidden_dim, dynamics.ctrl_dim), 
        #                                 torch.nn.Tanh())
        self.proj = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, dynamics.ctrl_dim), 
                                        torch.nn.Tanh())
        self.initialize_rnn_h = torch.nn.Sequential(torch.nn.Linear(8 * 8 + self.state_dim, hidden_dim),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(hidden_dim, hidden_dim),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(hidden_dim, hidden_dim))
        self.initialize_rnn_c = torch.nn.Sequential(torch.nn.Linear(8 * 8 + self.state_dim, hidden_dim),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(hidden_dim, hidden_dim),
                                                    torch.nn.Tanh(),
                                                    torch.nn.Linear(hidden_dim, hidden_dim))
        self.L2loss = torch.nn.MSELoss()
        self.leakyrelu = torch.nn.LeakyReLU()

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

    def update_env(self, center_x):
        final_x = self.env.final.center[0]
        obs_x = (center_x + final_x) / 2
        self.env.covers[0].center = [center_x, 3.5]
        self.env.obs[0].center = [obs_x, 9.0]

    def initial_rnn_state(self, imgs, x0):
        # x0 is [bs, 1, state_dim]
        y = self.cnn(imgs)    # [bs, 1, 8 8]
        if (imgs.shape[0] == 1) & (x0.shape[0] > 1):
            bs = x0.shape[0]
            y = y.repeat(bs, 1, 1, 1)
        y0 = torch.cat([y.view(*y.shape[:2], -1), x0], dim=-1).permute([1,0,2])
        return self.initialize_rnn_h(y0), self.initialize_rnn_c(y0)

    def forward(self, x0, imgs, h0=None):
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
            h0 = self.initial_rnn_state(imgs, x0)
            # if (imgs.shape[0] == 1) & (x0.shape[0] > 1):
            #     bs = x0.shape[0]
            #     h0 = (h0_[0].repeat(1, bs, 1), h0_[1].repeat(1, bs, 1))
            # else:
            #     h0 = h0_

        o, h0 = self.lstm(x0, h0)    # [bs, time_dim, hidden_dim] , bs = 1 for a single expert trajectory.

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
        return o, u, x_next, h0



    def propagate_n(self, n, x_partial, imgs, h0=None, time_dim=1):
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
        if h0 is None:
            h0_ = self.initial_rnn_state(imgs, x_partial)
            if (imgs.shape[0] == 1) & (x_partial.shape[0] > 1):
                bs = x_partial.shape[0]
                h0 = (h0_[0].repeat(1, bs, 1), h0_[1].repeat(1, bs, 1))
            else:
                h0 = h0_
        

        x_future = []
        u_future = []
        x_prev = x_partial[:,-1:,:]    # [bs, 1, state_dim]
        o, h = self.lstm(x_prev, h0)    # h is the last hidden state/last output
        # get last state, as that is the input to compute the first step of the n steps
        

        for i in range(n):
            u_ = self.proj(o[:,-1:,:])    # [bs, 1, ctrl_dim]
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


    def reconstruction_loss(self, x_true, u_true, imgs, tls, teacher_training=0.0, time_dim=1):
        '''
        Given an input trajectory x_traj, compute the reconstruction error for state and control.

        Inputs:
            x_true is [bs, time_dim, state_dim] --- input state trajectory (from expert demonstration)
            u_true is [bs, time_dim, ctrl_dim] --- input control trajectory (from expert demonstration)
            teacher_training in [0,1] --- a probability of using the previous propagated state as opposed to the true state from x_true
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
            o, u_pred, x_pred, h0 = self.forward(x_true, imgs)
            x_pred = self.join_partial_future_signal(x_true[:,:1,:], x_pred[:,:-1,:])
            return self.L2loss(x_pred, x_true), self.L2loss(u_pred, u_true), x_pred, u_pred
        else:
            prob = np.random.rand(x_true.shape[1]-1) < teacher_training
            xs = []
            us = []
            xs.append(x_true[:,:1,:])
            x_input = xs[-1]
            h0 = None
            for t in range(x_true.shape[1]-1):
                o, u, x_next, h0 = self.forward(x_input, imgs, h0=h0)
                xs.append(x_next)
                us.append(u)
                if prob[t]:
                    x_input = x_next
                else:
                    x_input = x_true[:,t+1:t+2,:]
            xx = torch.cat(xs, time_dim)
            o, u, _, _ = self.forward(x_input, imgs, h0=h0)
            us.append(u)
            uu = torch.cat(us, time_dim)
            
            d = (xx - x_true).pow(2).cumsum(1).mean(-1, keepdims=True)
            recon_state_loss = get_tl_elements(d, tls).mean()
            
            d = (uu - u_true).pow(2).cumsum(1).mean(-1, keepdims=True)
            recon_ctrl_loss = get_tl_elements(d, tls).mean()
            return recon_state_loss, recon_ctrl_loss, xx, uu


    @staticmethod
    def join_partial_future_signal( x_partial, x_future, time_dim=1):
        return torch.cat([x_partial, x_future], time_dim)

    def STL_robustness(self, x, formula, formula_input, **kwargs):
        signal = self.unstandardize_x(x)    # [bs, time_dim, state_dim]
        return formula.robustness(formula_input(signal, self.env), **kwargs)

    def STL_loss(self, x, formula, formula_input, **kwargs):
        # penalize negative robustness values only
        robustness = self.STL_robustness(x, formula, formula_input, **kwargs)
        return self.leakyrelu(-robustness).mean()
        # violations = robustness[robustness < 0]
        # if len(violations) == 0:
        #     return torch.relu(-robustness).mean()
        # return -violations.mean()

    def adversarial_STL_loss(self, x, formula, formula_input, **kwargs):
        # minimize mean robustness
        return self.STL_robustness(x, formula, formula_input, **kwargs).mean()
