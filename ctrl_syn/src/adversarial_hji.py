import sys
import torch
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
import IPython
from torch.utils.data import Dataset, DataLoader
from learning import *
from utils import *


draw_params = {"initial": {"color": "lightskyblue", "fill": True, "alpha": 0.5}, "final": {"color": "coral", "fill": True, "alpha": 0.5}, "covers": {"color": "black", "fill": False}, "obs": {"color": "red", "fill": True, "alpha": 0.5} }
vf_cpu =  HJIValueFunction.apply



def adversarial(model, T, formula, formula_input_func, device, tqdm, writer, hps, save_model_path, number, iter_max=50, adv_n_samples=32):
    log_dir = save_model_path.replace("models", "logs", 1)
    fig_dir = save_model_path.replace("models", "figs", 1)

    lr = hps.learning_rate
    alpha = hps.alpha
    print("\nAdversarial training on model:", save_model_path, "\n")
    checkpoint = torch.load(save_model_path + '/model')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    model = model.to(device)
    model.switch_device(device)
    mu_x, sigma_x = model.stats[0][:,:,:4], model.stats[1][:,:,:4]
    mu_u, sigma_u = model.stats[0][:,:,4:], model.stats[1][:,:,4:]

    ic_var_ =  torch.Tensor(InitialConditionDataset(adv_n_samples, vf_cpu, hps.stl_type)).float().permute([1,0,2]).to(device)            # [time, batch, x_dim]

    ic_var =  standardize_data(ic_var_, mu_x, sigma_x).to(device).requires_grad_(True)                          # [time, batch, x_dim]
    if hps.stl_type == "test":
        th_min = -np.pi/4
        th_max = np.pi/4
    elif hps.stl_type == "coverage_test":
        th_min = np.pi/2
        th_max = np.pi

    x_min = torch.tensor([(model.env.initial.lower[0] - mu_x[0,0,0])/sigma_x[0,0,0],
                          (model.env.initial.lower[1] - mu_x[0,0,1])/sigma_x[0,0,1],
                          (th_min - mu_x[0,0,2])/sigma_x[0,0,2],
                          -mu_x[0,0,3]/sigma_x[0,0,3]]).unsqueeze(0).unsqueeze(0).to(device)
    x_max = torch.tensor([(model.env.initial.upper[0] - mu_x[0,0,0])/sigma_x[0,0,0],
                          (model.env.initial.upper[1] - mu_x[0,0,1])/sigma_x[0,0,1],
                          (th_max - mu_x[0,0,2])/sigma_x[0,0,2],
                          (2 - mu_x[0,0,3])/sigma_x[0,0,3]]).unsqueeze(0).unsqueeze(0).to(device)

    with tqdm(total=iter_max) as pbar:
        fig, axes = plt.subplots(1, 3, figsize=(25,6))
        ic_var_un = unstandardize_data(ic_var, mu_x, sigma_x).cpu().detach().numpy()
        ax1 = axes[0]
        _, ax1 = plot_hji_contour(ax1)
        _, ax1 = model.env.draw2D(ax=ax1, kwargs=draw_params)
        ax1.scatter(ic_var_un[:,:,0], ic_var_un[:,:,1], s=50, c='k', zorder=10)
        ax1.set_xlim([-2., 2.])
        ax1.set_ylim([-2., 2.])

        ax2 = axes[1]
        ax2.fill([th_min, th_max, th_max, th_min, th_min], [0,0, 1, 1, 0], 'red', alpha=0.5)
        th_y = np.arange(ic_var_un[:,:,1].shape[-1])/ic_var_un[:,:,1].shape[-1]
        ax2.scatter(ic_var_un[:,:,2], th_y, s=50, c='black', zorder=10)
        ax2.grid()
        ax2.set_xlim([th_min - np.pi/4., th_max + np.pi/4])
        ax2.set_ylim([0., 1.])

        ax3 = axes[2]
        v_min = 0
        v_max = 2
        ax3.fill([v_min, v_max, v_max, v_min, v_min], [0, 0, 1, 1, 0], 'red', alpha=0.5)
        ax3.scatter(ic_var_un[:,:,3], th_y, s=50, c='black', zorder=10)
        ax3.grid()
        ax3.set_xlim([v_min - 0.5, v_max + 0.5])
        ax3.set_ylim([0., 1.])
        eps = 1E-4
        precision = 4
        for iteration in range(iter_max):
            x_future, u_future = model.propagate_n(T, ic_var)
            complete_traj = model.join_partial_future_signal(ic_var, x_future)
            loss_spc = model.adversarial_safety_preserving_loss(complete_traj[:-1,:,:], u_future)
            loss_hji = model.adversarial_HJI_loss(complete_traj)
            loss_stl = model.adversarial_STL_loss(complete_traj, formula, formula_input_func, scale=hps.adv_stl_scale(iteration))

            # loss_stl = model.adversarial_HJI_loss(complete_traj)
            loss = loss_spc + loss_hji + loss_stl

            loss.backward()
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
                # while (model.value_func(unstandardize_data(ic_var, mu_x, sigma_x)).relu().sum() > 0) | (((x_min > ic_var) | (x_max < ic_var)).any(-1, keepdims=True).sum() > 0):
                x1 = torch.clone(ic_var)
                # backstepping line search
                while ((((ic_var - x_min) < 0.0) | ((x_max - ic_var) < 0.0)).any(-1, keepdims=True).sum() > 0):
                    count += 1
                    # ic_var += torch.where((model.value_func(unstandardize_data(ic_var, mu_x, sigma_x)) > 0) | (((x_min > ic_var) | (x_max < ic_var)).any(-1, keepdims=True)), hps.alpha * gradient, torch.zeros_like(ic_var))
                    ic_var += torch.where((((ic_var - x_min) < 0.0) | ((x_max - ic_var) < 0.0)).any(-1, keepdims=True), hps.alpha * gradient, torch.zeros_like(ic_var))

                    if count > (lr // hps.alpha) * 1.5:
                        write_log(log_dir, "Adversarial: Line search took {} search steps.".format(count))

                        breakpoint()
                

                ic_var.grad.zero_()

            # plotting initial states as they change over each iteration
            ic_var_un = unstandardize_data(ic_var, mu_x, sigma_x).cpu().detach().numpy()
            ax1.scatter(ic_var_un[:,:,0], ic_var_un[:,:,1], s=25, c='grey', zorder=15)
            ax2.scatter(ic_var_un[:,:,2], th_y, s=25, c='grey', zorder=15)
            ax3.scatter(ic_var_un[:,:,3], th_y, s=25, c='grey', zorder=15)
            loss_stl_true = model.adversarial_STL_loss(complete_traj, formula, formula_input_func, scale=-1)
            writer.add_scalar('adversarial/stl_true', loss_stl_true, (iter_max * number) + iteration)
            writer.add_scalar('adversarial/stl_soft', loss_stl, (iter_max * number) + iteration)        
            writer.add_scalar('adversarial/spc', loss_spc, (iter_max * number) + iteration)
            writer.add_scalar('adversarial/hji', loss_hji, (iter_max * number) + iteration)
            writer.add_scalar('adversarial/loss_true', loss_spc + loss_stl_true + loss_hji, (iter_max * number) + iteration)
            writer.add_scalar('adversarial/loss_soft', loss, (iter_max * number) + iteration)
            writer.add_scalar('adversarial/stl_scale', hps.adv_stl_scale(iteration), (iter_max * number) + iteration)
            writer.add_figure('adversarial/initial_states', fig, (iter_max * number) + iteration)

            # plotting adversarial trajectories as the initial states change
            x_future, u_future = model.propagate_n(T, ic_var)
            complete_traj = model.join_partial_future_signal(ic_var, x_future)
            traj_np = unstandardize_data(complete_traj, mu_x, sigma_x).cpu().detach().numpy()

            fig1, ax = plt.subplots(figsize=(10,10))
            _, ax = model.env.draw2D(ax=ax, kwargs=draw_params)
            ax.axis("equal")
            _, ax = plot_hji_contour(ax)
            for j in range(traj_np.shape[1]):
                ax.plot(traj_np[:,j,0], traj_np[:,j,1])
                ax.scatter(traj_np[:,j,0], traj_np[:,j,1])

            ax.set_xlim([-5, 15])
            ax.set_ylim([-2, 12])
            fig1.savefig(fig_dir + '/adversarial/number={:02d}_iteration={:03d}.png'.format(number, (iter_max * number) + iteration))

            writer.add_figure('adversarial/trajectory', fig1, iteration)

            pbar.set_postfix(loss='{:.2e}'.format(loss))
            pbar.update(1)

    hji = model.HJI_value(complete_traj).squeeze()
    ham = model.hamiltonian(complete_traj[:-1,:,:], u_future).squeeze()
    stl = model.STL_robustness(complete_traj, formula, formula_input_func)

    violating_idx = (hji > 0).any(0) | ((ham < 0) & (hji[:-1,:] < 0)).any(0) | (stl.squeeze() < 0)

    adv_ic = ic_var[:,violating_idx,:] # [1, batch, x_dim]
    if violating_idx.sum() == 0.0:
        breakpoint()
        return adv_ic
    ret_adv_ic = unstandardize_data(adv_ic, mu_x, sigma_x).to("cpu").detach().permute([1,0,2])
    return ret_adv_ic




