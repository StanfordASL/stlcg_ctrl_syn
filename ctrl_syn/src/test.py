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

import argparse

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
            ax.plot(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], linewidth=4, c='k', linestyle='--', zorder=2)
            ax.scatter(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], s=100, c='k', label="Expert", zorder=3)
            # plotting propagated expert trajectory
            ax.plot(x_traj_prop[:,0], x_traj_prop[:,1], linewidth=3, c="dodgerblue", linestyle='--', label="Reconstruction", zorder=4)
            ax.scatter(x_traj_prop[:,0], x_traj_prop[:,1], s=100, c="dodgerblue", zorder=5)
            # plotting predicted expert trajectory during training (with teacher training)
            ax.plot(x_traj_pred.cpu().detach().squeeze().numpy()[:,0], x_traj_pred.cpu().detach().squeeze().numpy()[:,1], linewidth=3, c="IndianRed", linestyle='--', label="Expert recon.", zorder=6)
            ax.scatter(x_traj_pred.cpu().detach().squeeze().numpy()[:,0], x_traj_pred.cpu().detach().squeeze().numpy()[:,1], s=100, c="IndianRed", zorder=7)

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



def adversarial(model, T, formula, formula_input_func, device, save_model_path, lower, upper, iter_max=50, adv_n_samples=32):
    lr = 0.1
    alpha = 0.001
    print("\nAdversarial training on model:", save_model_path, "\n")
    checkpoint = torch.load(save_model_path + '/model')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    model = model.to(device)
    model.switch_device(device)
    mu_x, sigma_x = model.stats[0][:,:,:4], model.stats[1][:,:,:4]
    mu_u, sigma_u = model.stats[0][:,:,4:], model.stats[1][:,:,4:]

    env = model.env
    ic_var_ = initial_conditions(adv_n_samples, lower, upper)

    ic_var =  model.standardize_x(ic_var_).to(device).requires_grad_(True)                          # [batch, time, x_dim]

    x_min = model.standardize_x(lower).to(device)
    x_max = model.standardize_x(upper).to(device)

    fig, axes = plt.subplots(1, 3, figsize=(25,6))
    ic_var_un = model.unstandardize_x(ic_var).cpu().detach().numpy()
    ax1 = axes[0]
    _, ax1 = model.env.draw2D(ax=ax1, kwargs=draw_params)
    ax1.scatter(ic_var_un[:,:,0], ic_var_un[:,:,1], s=50, c='k', zorder=10)
    ax1.set_xlim([lower[0].numpy() - 0.5, upper[0].numpy() + 0.5])
    ax1.set_ylim([lower[1].numpy() - 0.5, upper[1].numpy() + 0.5])

    ax2 = axes[1]
    th_min = lower[2].numpy()
    th_max = upper[2].numpy()
    ax2.fill([th_min, th_max, th_max, th_min, th_min], [0,0, 1, 1, 0], 'red', alpha=0.5)
    th_y = np.arange(adv_n_samples)/adv_n_samples
    ax2.scatter(ic_var_un[:,:,2], th_y, s=50, c='black', zorder=10)
    ax2.grid()
    ax2.set_xlim([th_min - np.pi/4., th_max + np.pi/4])
    ax2.set_ylim([0., 1.])

    ax3 = axes[2]
    v_min = lower[3].numpy()
    v_max = upper[3].numpy()
    ax3.fill([v_min, v_max, v_max, v_min, v_min], [0, 0, 1, 1, 0], 'red', alpha=0.5)
    ax3.scatter(ic_var_un[:,:,3], th_y, s=50, c='black', zorder=10)
    ax3.grid()
    ax3.set_xlim([v_min - 0.5, v_max + 0.5])
    ax3.set_ylim([0., 1.])

    eps = 1E-4
    precision = 4

    print("starting adversarial training loop")
    for iteration in range(iter_max):
        x_future, u_future = model.propagate_n(T, ic_var)
        complete_traj = model.join_partial_future_signal(ic_var, x_future)
        loss_stl = model.adversarial_STL_loss(complete_traj, formula, formula_input_func, scale=1)
        loss_stl.backward()

        with torch.no_grad():
            # make gradient step
            gradient = (ic_var.grad * 10**precision).round() / 10**precision
            x0 = torch.clone(ic_var)
            x1 = torch.clone(ic_var - lr * gradient)
            # on the boundary + eps or outside the box & the gradient step will take it outside the box, then do not step
            ic_var -= torch.where((((ic_var - x_min) <= eps) | ((x_max - ic_var) <= eps)).any(-1, keepdims=True) & (((x1 - x_min) <= eps) | ((x_max - x1) <= eps)).any(-1, keepdims=True) , torch.zeros_like(ic_var), lr * gradient)
            # project back into BRS
            count = 0
            # if any of the ic_var are outside of the BRS or outside the initial set
            x1 = torch.clone(ic_var)
            # backstepping line search
            while ((((ic_var - x_min) < 0.0) | ((x_max - ic_var) < 0.0)).any(-1, keepdims=True).sum() > 0):
                count += 1

                ic_var += torch.where((((ic_var - x_min) < 0.0) | ((x_max - ic_var) < 0.0)).any(-1, keepdims=True), alpha * gradient, torch.zeros_like(ic_var))

                if count > (lr // alpha) * 1.5:
                    write_log(log_dir, "Adversarial: Line search took {} search steps.".format(count))
                    breakpoint()


            ic_var.grad.zero_()
        print("Iteration {}".format(iteration))


        # plotting initial states as they change over each iteration
        ic_var_un = model.unstandardize_x(ic_var).cpu().detach().numpy()
        ax1.scatter(ic_var_un[:,:,0], ic_var_un[:,:,1], s=25, c='grey', zorder=15)
        ax2.scatter(ic_var_un[:,:,2], th_y, s=25, c='grey', zorder=15)
        ax3.scatter(ic_var_un[:,:,3], th_y, s=25, c='grey', zorder=15)
        loss_stl_true = model.adversarial_STL_loss(complete_traj, formula, formula_input_func, scale=-1)

        # plotting adversarial trajectories as the initial states change
        x_future, u_future = model.propagate_n(T, ic_var)
        complete_traj = model.join_partial_future_signal(ic_var, x_future)
        traj_np = model.unstandardize_x(complete_traj).cpu().detach().numpy()

        fig1, ax = plt.subplots(figsize=(10,10))
        _, ax = model.env.draw2D(ax=ax, kwargs=draw_params)
        # plotting the sampled initial state trajectories
        ax.plot(traj_np[:,:,0].T, traj_np[:,:,1].T, alpha=0.5, c='RoyalBlue')
        ax.scatter(traj_np[:,:,0].T, traj_np[:,:,1].T, alpha=0.5, c='RoyalBlue')

        ax.set_xlim([-5, 15])
        ax.set_ylim([-5, 15])
        ax.axis("equal")

        fig1.savefig("../figs/test/adversarial_traj_{:03d}.png".format( iteration))

    fig.savefig("../figs/test/adversarial_ic_{:03d}.png".format( iteration))


    stl = model.STL_robustness(complete_traj, formula, formula_input_func)
    print(stl.squeeze().detach().cpu().numpy())
    violating_idx = stl.squeeze() < 0


    if violating_idx.sum() == 0.0:
        print("No violating initial conditions")
        return adv_ic
    adv_ic = ic_var[violating_idx,:,:] # [batch, 1, x_dim]
    ret_adv_ic = model.unstandardize_x(adv_ic).to("cpu").detach()
    return ret_adv_ic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type',  type=str, default="train", help="test train or adversarial code")

    args = parser.parse_args()

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
    hidden_dim = 64
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

    if args.type == "train":
        test(model, (x_train, u_train), formula, formula_input_func, train_loader, device)
    elif args.type == "adversarial":
        save_model_path = "../models/coverage_run=1_lstm_dim=64_teacher_training=-1.0_weight_ctrl=0.5_weight_recon=1.0_weight_stl=-1.0_stl_max=0.2_stl_min=0.10_stl_scale=-1.0_scale_max=50.0_scale_min=0.10_iter_max=500.0"
        adversarial(model, x_train_.shape[1]+10, formula, formula_input_func, device, save_model_path, lower, upper, iter_max=50, adv_n_samples=32)