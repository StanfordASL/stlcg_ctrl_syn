import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from learning import *
from stl import *


sys.path.append('../../../stlcg_karen/src')
sys.path.append('../../expert_demo_ros/src/utils')

import stlcg
import stlviz
from environment import *

draw_params = {"initial": {"color": "lightskyblue", "fill": True, "alpha": 0.5}, "final": {"color": "coral", "fill": True, "alpha": 0.5}, "covers": {"color": "black", "fill": False}, "obs": {"color": "red", "fill": True, "alpha": 0.5} }


def test(model, train_traj, formula, formula_input_func, train_loader, device):
    '''
    Function to test if all the functions are working as expected.

    Inputs:
    model              : The neural network model (refer to learning.py)               
    train_traj         : Expert trajectory training set             
    formula            : STL specification to be satisfied  
    formula_input_func : STL input function corresponding to STL specification
    train_loader       : PyTorch data loader for initial condition train set
    device             : cpu or cuda
    '''


    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.1)

    # switch everything to the specified device
    x_train, u_train = train_traj[0].to(device), train_traj[1].to(device)
    model = model.to(device)
    model.switch_device(device)

    x_train_ = model.unstandardize_x(x_train)
    u_train_ = model.unstandardize_u(u_train)

    T = x_train.shape[1]

    for (batch_idx, ic_) in enumerate(train_loader):
        
        optimizer.zero_grad()
        ic = ic_.to(device).float()        # [bs, 1, x_dim]
        model.train()
        # parameters
        teacher_training_value = 0.5
        weight_stl = 0.5
        stl_scale_value = 0.5

        # reconstruct the expert model
        loss_state, loss_ctrl, x_traj_pred, u_traj_pred = model.reconstruction_loss(x_train, u_train, teacher_training=teacher_training_value)
        loss_recon = loss_state + 0.5 * loss_ctrl

        # with new ICs, propagate the trajectories
        x_future, u_future = model.propagate_n(T, ic)
        complete_traj = model.join_partial_future_signal(ic, x_future)      # [bs, time_dim, x_dim]

        # stl loss
        loss_stl = model.STL_loss(complete_traj, formula, formula_input_func, scale=stl_scale_value)
        loss_stl_true = model.STL_loss(complete_traj, formula, formula_input_func, scale=-1)

        # total loss
        loss = loss_recon + weight_stl * loss_stl
        
        # plotting progress
        if batch_idx % 20 == 0:
            # trajectories from propagating initial states
            traj_np = model.unstandardize_x(complete_traj).cpu().detach().numpy()
            # trajectory from propagating initial state of expert trajectory
            x_future, u_future = model.propagate_n(T, x_train[:,:1,:])
            x_traj_prop = model.join_partial_future_signal(x_train[:,:1,:], x_future)
            x_traj_prop = model.unstandardize_x(x_traj_prop).squeeze().detach().cpu().numpy()
            # trajectory from teacher training, and used for reconstruction loss (what the training sees)
            x_traj_pred = model.unstandardize_x(x_traj_pred)

            # trajectory plot
            fig1, ax = plt.subplots(figsize=(10,10))
            _, ax = model.env.draw2D(ax=ax, kwargs=draw_params)
            ax.axis("equal")

            # plotting the sampled initial state trajectories
            ax.plot(traj_np[:,:,0].T, traj_np[:,:,1].T, alpha=0.4, c='RoyalBlue')
            ax.scatter(traj_np[:,:,0].T, traj_np[:,:,1].T, alpha=0.4, c='RoyalBlue')
            # plotting true expert trajectory
            ax.plot(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], linewidth=4, c='k', linestyle='--')
            ax.scatter(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], s=100, c='k', label="Expert")
            # plotting propagated expert trajectory
            ax.plot(x_traj_prop[:,0], x_traj_prop[:,1], linewidth=3, c="dodgerblue", linestyle='--', label="Reconstruction")
            ax.scatter(x_traj_prop[:,0], x_traj_prop[:,1], s=100, c="dodgerblue")
            # plotting predicted expert trajectory during training (with teacher training)
            ax.plot(x_traj_pred.cpu().detach().squeeze().numpy()[:,0], x_traj_pred.cpu().detach().squeeze().numpy()[:,1], linewidth=3, c="IndianRed", linestyle='--', label="Expert recon.")
            ax.scatter(x_traj_pred.cpu().detach().squeeze().numpy()[:,0], x_traj_pred.cpu().detach().squeeze().numpy()[:,1], s=100, c="IndianRed")

            ax.set_xlim([-5, 15])
            ax.set_ylim([-2, 12])
            fig1.savefig("../figs/test/trajectories_{:03d}.png".format(batch_idx))

            # controls plot
            fig2, axs = plt.subplots(1,2,figsize=(15,6))
            for (k,a) in enumerate(axs):
                a.plot(u_train_.squeeze().cpu().detach().numpy()[:,k], label="Expert")
                a.plot(model.unstandardize_u(u_future).squeeze().cpu().detach().numpy()[:,k], linestyle='--', label="Reconstruction")
                a.grid()
                a.set_xlim([0,T])
                a.set_ylim([-4,4])
            fig2.savefig("../figs/test/controls_{:03d}.png".format(batch_idx))

        loss.backward()
        optimizer.step()


if __name__ == "__main__":

    env = { "covers": [Circle([8., 3.0], 2.0)],
           "obstacles": [Circle([4.5, 6.], 1.5)],
           "initial": Box([0., 0.],[3., 3.]),
           "final": Circle([1., 9.], 1.0)
            }   
    env = Environment(env)
    device = "cpu"
    x_train_, u_train_, stats = prepare_data("../expert/coverage/train.npy")
    x_train = standardize_data(x_train_, stats[0][:,:,:4], stats[1][:,:,:4])
    u_train = standardize_data(u_train_, stats[0][:,:,4:], stats[1][:,:,4:])
    dynamics = KinematicBicycle(0.5)
    hidden_dim = 32
    model = STLPolicy(dynamics, hidden_dim, stats, env, dropout=0., num_layers=1)


    # initial conditions set
    lower = torch.tensor([env.initial.lower[0], env.initial.lower[1], -np.pi/4, 0])
    upper = torch.tensor([env.initial.upper[0], env.initial.upper[1], np.pi/4, 3])

    ic_ = initial_conditions(128, lower, upper)
    ic = model.standardize_x(ic_)
    train_loader = torch.utils.data.DataLoader(ic, batch_size=128//32, shuffle=True)


    in_end_goal = inside_circle(env.final, "distance to final")
    stop_in_end_goal = in_end_goal & (stlcg.Expression('speed')  < 0.5)
    end_goal = stlcg.Eventually(subformula=stop_in_end_goal)
    coverage = stlcg.Eventually(subformula=(always_inside_circle(env.covers[0], "distance to coverage", interval=[0,10]) & (stlcg.Expression('speed')  < 1.0)))
    avoid_obs = always_outside_circle(env.obs[0], "distance to obstacle")

    formula = stlcg.Until(subformula1=coverage, subformula2=end_goal) & avoid_obs
    stl_graph = stlviz.make_stl_graph(formula)
    stlviz.save_graph(stl_graph, "../figs/test/stl")
    formula_input_func=lambda s: get_formula_input(s, env.covers[0], env.obs[0], env.final, device, backwards=False)

    test(model, (x_train, u_train), formula, formula_input_func, train_loader, device)