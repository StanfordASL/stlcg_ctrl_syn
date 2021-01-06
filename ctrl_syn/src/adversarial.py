import sys
sys.path.append('src')
import os
import torch
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
import IPython
from torch.utils.data import Dataset, DataLoader
from learning import *
from utils import *

pardir = os.path.dirname(os.path.dirname(__file__))


draw_params = {"initial": {"color": "lightskyblue", "fill": True, "alpha": 0.5}, "final": {"color": "coral", "fill": True, "alpha": 0.5}, "covers": {"color": "black", "fill": False}, "obs": {"color": "red", "fill": True, "alpha": 0.5} }


# adversarial using gradient descent
def adversarial_gd(model, T, formula, formula_input_func, device, tqdm, writer, hps, save_model_path, number, lower, upper, iter_max=50, adv_n_samples=128):
    log_dir = save_model_path.replace("models", "logs", 1)
    fig_dir = save_model_path.replace("models", "figs", 1)

    lr = hps.learning_rate

    print("\nAdversarial training on model:", save_model_path, "\n")
    checkpoint = torch.load(save_model_path + '/model')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    model = model.to(device)
    model.switch_device(device)

    env = model.env
    ic_var_ = initial_conditions(adv_n_samples, lower, upper).to(device)
    ic_var =  model.standardize_x(ic_var_).to(device).requires_grad_(True)                          # [batch, time, x_dim]

    x_min = model.standardize_x(lower.to(device))
    x_max = model.standardize_x(upper.to(device))

    with tqdm(total=iter_max) as pbar:
        fig, axes = plt.subplots(1, 3, figsize=(25,6))
        ic_var_un = model.unstandardize_x(ic_var).cpu().detach().numpy()
        ax1 = axes[0]
        _, ax1 = model.env.draw2D(ax=ax1, kwargs=draw_params)
        ax1.scatter(ic_var_un[:,:,0], ic_var_un[:,:,1], s=50, c='k', zorder=10, alpha=0.3)
        ax1.set_xlim([lower[0].numpy() - 0.5, upper[0].numpy() + 0.5])
        ax1.set_ylim([lower[1].numpy() - 0.5, upper[1].numpy() + 0.5])

        ax2 = axes[1]
        th_min = lower[2].numpy()
        th_max = upper[2].numpy()
        ax2.fill([th_min, th_max, th_max, th_min, th_min], [0,0, 1, 1, 0], 'red', alpha=0.3)
        th_y = np.arange(adv_n_samples)/adv_n_samples
        ax2.scatter(ic_var_un[:,:,2], th_y, s=50, c='black', zorder=10)
        ax2.grid()
        ax2.set_xlim([th_min - np.pi/4., th_max + np.pi/4])
        ax2.set_ylim([0., 1.])

        ax3 = axes[2]
        v_min = lower[3].numpy()
        v_max = upper[3].numpy()
        ax3.fill([v_min, v_max, v_max, v_min, v_min], [0, 0, 1, 1, 0], 'red', alpha=0.3)
        ax3.scatter(ic_var_un[:,:,3], th_y, s=50, c='black', zorder=10)
        ax3.grid()
        ax3.set_xlim([v_min - 0.5, v_max + 0.5])
        ax3.set_ylim([0., 1.])

        eps = 1E-4
        precision = 4

        for iteration in range(iter_max):
            x_future, u_future = model.propagate_n(T, ic_var)
            complete_traj = model.join_partial_future_signal(ic_var, x_future)
            loss_stl = model.adversarial_STL_loss(complete_traj, formula, formula_input_func, scale=hps.adv_stl_scale(iteration))
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
            ic_var_un = model.unstandardize_x(ic_var).cpu().detach().numpy()
            ax1.scatter(ic_var_un[:,:,0], ic_var_un[:,:,1], s=25, c='grey', zorder=15)
            ax2.scatter(ic_var_un[:,:,2], th_y, s=25, c='grey', zorder=15)
            ax3.scatter(ic_var_un[:,:,3], th_y, s=25, c='grey', zorder=15)
            fig.savefig(fig_dir + '/adversarial/ic_number={:02d}_iteration={:03d}.png'.format(number, (iter_max * number) + iteration))

            loss_stl_true = model.adversarial_STL_loss(complete_traj, formula, formula_input_func, scale=-1)
            writer.add_scalar('adversarial/stl_true', loss_stl_true, (iter_max * number) + iteration)
            writer.add_scalar('adversarial/stl_soft', loss_stl, (iter_max * number) + iteration)
            writer.add_scalar('adversarial/stl_scale', hps.adv_stl_scale(iteration), (iter_max * number) + iteration)
            writer.add_figure('adversarial/initial_states', fig, (iter_max * number) + iteration)

            # plotting adversarial trajectories as the initial states change
            x_future, u_future = model.propagate_n(T, ic_var)
            complete_traj = model.join_partial_future_signal(ic_var, x_future)
            traj_np = model.unstandardize_x(complete_traj).cpu().detach().numpy()

            fig1, ax = plt.subplots(figsize=(10,10))
            _, ax = model.env.draw2D(ax=ax, kwargs=draw_params)
            ax.axis("equal")
            # plotting the sampled initial state trajectories
            ax.plot(traj_np[:,:,0].T, traj_np[:,:,1].T, alpha=0.5, c='RoyalBlue')
            ax.scatter(traj_np[:,:,0].T, traj_np[:,:,1].T, alpha=0.5, c='RoyalBlue')

            ax.set_xlim([-5, 15])
            ax.set_ylim([-5, 15])
            fig1.savefig(fig_dir + '/adversarial/traj_number={:02d}_iteration={:03d}.png'.format(number, (iter_max * number) + iteration))

            writer.add_figure('adversarial/trajectory', fig1, iteration)

            pbar.set_postfix(loss='{:.2e}'.format(loss_stl))
            pbar.update(1)

    stl = model.STL_robustness(complete_traj, formula, formula_input_func)

    violating_idx = stl.squeeze() < 0


    if violating_idx.sum() == 0.0:
        print("No violating initial conditions")
        return adv_ic
    adv_ic = ic_var[violating_idx,:,:] # [batch, 1, x_dim]
    ret_adv_ic = model.unstandardize_x(adv_ic).to("cpu").detach()
    return ret_adv_ic


# rejection-acceptance sampling
def adversarial_rejacc(model, T, formula, formula_input_func, device, hps, save_model_path, number, lower, upper, adv_n_samples=512):
    log_dir = save_model_path.replace("models", "logs", 1)
    fig_dir = save_model_path.replace("models", "figs", 1)

    print("\nAdversarial training on model:", save_model_path, "\n")
    checkpoint = torch.load(save_model_path + '/model')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    model = model.to(device)
    model.switch_device(device)

    n = 10**4
    x0_ = initial_conditions(n, lower, upper).float()
    x0 = model.standardize_x(x0_)
    x_future, u_future = model.propagate_n(T, x0)
    complete_traj = model.join_partial_future_signal(x0, x_future)

    rho = model.STL_robustness(complete_traj, formula, formula_input_func, scale=-1).squeeze()

    violating_idx = rho <= 0
    satisfying_idx = rho > 0

    traj_np = model.unstandardize_x(complete_traj).cpu().detach().numpy()


    fig1, ax = plt.subplots(figsize=(10,10))
    _, ax = model.env.draw2D(ax=ax, kwargs=draw_params)
    ax.axis("equal")
    # plotting the sampled initial state trajectories
    ax.plot(traj_np[satisfying_idx,:,0].T, traj_np[satisfying_idx,:,1].T, alpha=0.5, c='RoyalBlue')
    ax.scatter(traj_np[satisfying_idx,:,0].T, traj_np[satisfying_idx,:,1].T, alpha=0.5, c='RoyalBlue')
    ax.plot(traj_np[violating_idx,:,0].T, traj_np[violating_idx,:,1].T, alpha=0.5, c='IndianRed')
    ax.scatter(traj_np[violating_idx,:,0].T, traj_np[violating_idx,:,1].T, alpha=0.5, c='IndianRed', marker="x")

    ax.set_xlim([-5, 15])
    ax.set_ylim([-5, 15])

    fig1.savefig(fig_dir + '/adversarial/traj_number={:02d}.png'.format(number))

    writer.add_figure('adversarial/trajectory', fig1, number)
    plt.close(fig1)





# rejection-acceptance sampling
def adversarial_rejacc_cnn(model, T, formula, formula_input_func, device, hps, save_model_path, number, lower, upper, adv_n_samples=512):
    log_dir = save_model_path.replace("models", "logs", 1)
    fig_dir = save_model_path.replace("models", "figs", 1)

    print("\nAdversarial training on model:", save_model_path, "\n")
    checkpoint = torch.load(save_model_path + '/model')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    model = model.to(device)
    model.switch_device(device)
    env_tmp = copy.deepcopy(model.env)

    # rejection-acceptance sampling

    x0_ = initial_conditions(adv_n_samples, lower, upper).float()
    x0 = model.standardize_x(x0_)

    # with ICs, propagate the trajectories
    xdim = x0.shape[-1]
    img_bs = hps.img_bs
    centers_ic = np.round(1+np.random.rand(img_bs) * 9, 1)    # between 1-10, and one decimal place
    # GROSS -- hard coding some parameters here :/ 
    final_x = model.env.final.center[0]
    obs_x = (centers_ic + final_x) / 2
    model.env.covers[0].center = torch.tensor(np.stack([centers_ic, 3.5 * np.ones_like(centers_ic)], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, adv_n_samples, 1, 1]).view([-1, 1, 2])
    model.env.obs[0].center = torch.tensor(np.stack([obs_x, 9. * np.ones_like(obs_x)], axis=1)).unsqueeze(1).unsqueeze(1).repeat([1, adv_n_samples, 1, 1]).view([-1, 1, 2])

    ic_imgs = torch.cat([convert_env_img_to_tensor(os.path.join(pardir, "figs/environments/%.1f"%cb)) for cb in centers_ic], dim=0).to(device)

    ic_batch = x0[None,...].repeat([img_bs,1,1,1]).view(-1, 1, xdim)
    img_batch = ic_imgs.unsqueeze(1).repeat([1, adv_n_samples, 1, 1, 1]).view([-1, *ic_imgs.shape[-3:]])

    x_future, u_future = model.propagate_n(T, ic_batch, img_batch)
    complete_traj = model.join_partial_future_signal(ic_batch, x_future)      # [bs, time_dim, x_dim]

    rho = model.STL_robustness(complete_traj, formula, formula_input_func, scale=-1).squeeze()


    violating_idx = (rho <= 0).view(img_bs, adv_n_samples)
    satisfying_idx = (rho > 0).view(img_bs, adv_n_samples)

    traj_np = model.unstandardize_x(complete_traj).view(img_bs, adv_n_samples, T+1, xdim).cpu().detach()

    # trajectory plot
    fig1, ax1s = plt.subplots(2, img_bs//2, figsize=(20,10))
    cover_center_list = model.env.covers[0].center.view(img_bs, adv_n_samples, 1, 2)[:,0]
    obs_center_list = model.env.obs[0].center.view(img_bs, adv_n_samples, 1, 2)[:,0]

    for i in range(img_bs):
        env_tmp.covers[0].center = list(cover_center_list[i].squeeze().numpy())
        env_tmp.obs[0].center = list(obs_center_list[i].squeeze().numpy())

        ax = ax1s[i//(img_bs//2), i % (img_bs//2)]
        ax.grid(zorder=0)
        _, ax = env_tmp.draw2D(ax=ax, kwargs=draw_params)
        ax.axis("equal")
        ax.set_title("%i: cover_x = %.1f"%(i+1, env_tmp.covers[0].center[0]))
        ax.plot(traj_np[i,satisfying_idx[i],:,0].T, traj_np[i,satisfying_idx[i],:,1].T, alpha=0.4, c='RoyalBlue', zorder=5)
        ax.scatter(traj_np[i,satisfying_idx[i],:,0].T, traj_np[i,satisfying_idx[i],:,1].T, alpha=0.4, c='RoyalBlue', zorder=5)
        ax.plot(traj_np[i,violating_idx[i],:,0].T, traj_np[i,violating_idx[i],:,1].T, alpha=0.4, c='IndianRed', zorder=5)
        ax.scatter(traj_np[i,violating_idx[i],:,0].T, traj_np[i,violating_idx[i],:,1].T, alpha=0.4, c='IndianRed', zorder=5, marker="x")
        ax.set_xlim([-5, 15])
        ax.set_ylim([-5, 15])



    fig1.savefig(fig_dir + '/adversarial/traj_number={:02d}.png'.format(number))

    # writer.add_figure('adversarial/trajectory', fig1, number)
    plt.close(fig1)


    violating_idx = violating_idx.view(-1)

    return ic_batch[violating_idx], img_batch[violating_idx]

