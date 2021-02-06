import sys
import torch
import os
import numpy as np
import copy
from torch import nn, optim
import matplotlib.pyplot as plt
import IPython
from torch.utils.data import Dataset, DataLoader
from learning import *
from utils import *


pardir = os.path.dirname(os.path.dirname(__file__))



draw_params = {"initial": {"color": "lightskyblue", "fill": True, "alpha": 0.5}, "final": {"color": "coral", "fill": True, "alpha": 0.5}, "covers": {"color": "black", "fill": False}, "obs": {"color": "red", "fill": True, "alpha": 0.5} }





def train_cnn(case, model, train_traj, eval_traj, imgs, tls, env_param, formula, formula_input_func, train_loader, eval_loader, device, tqdm, writer, hps, save_model_path, number, iter_max=np.inf, status="new", xlim=[-6, 16], ylim=[-6, 16], plot_freq=10):
    '''
    This controls the control logic of the gradient steps

    Inputs:
    case               : coverage or drive scenario
    model              : The neural network model (refer to learning.py)               
    train_traj         : Expert trajectory training set             
    eval_traj          : Expert trajectory evaluation set         
    formula            : STL specification to be satisfied  
    formula_input_func : STL input function corresponding to STL specification
    train_loader       : PyTorch data loader for initial condition train set
    eval_loader        : PyTorch data loader for initial condition eval set
    device             : cpu or cuda
    tqdm               : progress bar object
    writer             : Tensorboard writer object
    hps                : hyperparameters for training (refer to run.py)
    save_model_path    : path where the model is saved
    number             : iteration for the training-adversarial loop
    iter_max           : maximum number of training epochs
    '''
    log_dir = save_model_path.replace("models", "logs", 1)
    fig_dir = save_model_path.replace("models", "figs", 1)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=hps.weight_decay)
    model_name = save_model_path.split("/")[-1]
    
    env_tmp = copy.deepcopy(model.env)
    expert_env = copy.deepcopy(model.env)

    # if training from scratch, initialize training steps at 0
    if status == "new":
        train_iteration = 0 # train iteration number
        eval_iteration = 0 # evaluation iteration number
        gradient_step = 0 # descent number
    # if continuing training the model, load the last iteration values
    elif status == "continue":
        print("\nContinue training model:", save_model_path, "\n")
        checkpoint = torch.load(save_model_path + "/model")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_iteration =  checkpoint['train_iteration']
        gradient_step =  checkpoint['gradient_step']
        eval_iteration =  checkpoint['eval_iteration']

    # switch everything to the specified device
    x_train, u_train = train_traj[0].to(device), train_traj[1].to(device)
    x_eval, u_eval = eval_traj[0].to(device), eval_traj[1].to(device)

    imgs_train, imgs_eval = imgs
    env_param_train, env_param_eval = env_param
    imgs_train = imgs_train.to(device)
    imgs_eval = imgs_eval.to(device)
    tls_train, tls_eval = tls
    tls_train = tls_train.to(device)
    tls_eval = tls_eval.to(device)
    model = model.to(device)
    model.switch_device(device)

    x_train_ = model.unstandardize_x(x_train)
    u_train_ = model.unstandardize_u(u_train)
    x_eval_ = model.unstandardize_x(x_eval)
    u_eval_ = model.unstandardize_u(u_eval)

    x_true_train = x_train_.view([-1,hps.expert_mini_bs, *x_train_.shape[-2:]]).cpu()
    u_true_train = u_train_.view([-1,hps.expert_mini_bs, *u_train_.shape[-2:]]).cpu()

    x_true_eval = x_eval_.view([-1,hps.expert_mini_bs, *x_eval_.shape[-2:]]).cpu()
    u_true_eval = u_eval_.view([-1,hps.expert_mini_bs, *u_eval_.shape[-2:]]).cpu()

    T = x_train.shape[1] + 10

    time = torch.arange(T).unsqueeze(-1).repeat(1, hps.expert_mini_bs) * model.dt
    # environments and images used for plotting. Computing here to avoid repeated computation in the loop
    env_param_ic_plotting = np.round(np.arange(2,10, dtype=float) + 0.5, 1) if case == "coverage" else sample_environment_parameters(case, 2)
    ic_imgs_plotting = torch.stack([generate_img_tensor_from_parameter(case, pi) for pi in env_param_ic_plotting]).to(device)

    carlength = 1.2
    height = 2

    with tqdm(total=(iter_max - train_iteration)) as pbar:
        while True:
            if train_iteration == iter_max:
                torch.save({
                           'train_iteration': train_iteration,
                           'gradient_step': gradient_step,
                           'eval_iteration': eval_iteration,
                           'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           'loss': loss,
                           'ic': ic},
                           save_model_path + "/model")

                torch.save({
                           'train_iteration': train_iteration,
                           'gradient_step': gradient_step,
                           'eval_iteration': eval_iteration,
                           'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           'loss': loss,
                           'ic': ic},
                           save_model_path + "/model_{:02d}".format(number))
                return

            hps_idx = train_iteration % (iter_max // (number + 1))
            for (batch_idx, ic_) in enumerate(train_loader):


                optimizer.zero_grad()
                ic = ic_["ic"].to(device).float()        # [bs, 1, x_dim]
                if "adv_ic" in ic_.keys():
                    adv_ic = ic_["adv_ic"].to(device).float()
                    adv_img_p =  ic_["adv_img_p"].float()
                    adv_samples = True
                else:
                    adv_samples = None


                model.train()
                # parameters
                teacher_training_value = hps.teacher_training(hps_idx)
                weight_stl = hps.weight_stl(hps_idx)
                stl_scale_value = hps.stl_scale(hps_idx)

                # reconstruct the expert model
                loss_state, loss_ctrl, x_traj_pred, u_traj_pred = model.reconstruction_loss(x_train, u_train, imgs_train, tls_train, teacher_training=teacher_training_value)
                loss_recon = loss_state + hps.weight_ctrl * loss_ctrl

                # with new ICs, propagate the trajectories
                ic_bs = ic.shape[0]
                xdim = ic.shape[-1]
                img_bs = hps.img_bs

                ps = sample_and_update_environment(case, model.env, img_bs, ic_bs)
                img_bs = ps.shape[0]

                ic_imgs = torch.stack([generate_img_tensor_from_parameter(case, pi) for pi in ps]).to(device)
                ic_batch = ic[None,...].repeat([img_bs,1,1,1]).view(-1, 1, xdim)
                img_batch = ic_imgs.unsqueeze(1).repeat([1, ic_bs, 1, 1, 1]).view([-1, *ic_imgs.shape[-3:]])

                if adv_samples is not None:
                    append_environment(case, model.env, adv_img_p, 1, carlength=carlength)
                    ic_batch = torch.cat([ic_batch, adv_ic], dim=0)
                    img_batch = torch.cat([img_batch, torch.stack([generate_img_tensor_from_parameter(case, pi) for pi in adv_img_p]).to(device)], dim=0)

                x_future, u_future = model.propagate_n(T, ic_batch, img_batch)
                complete_traj = model.join_partial_future_signal(ic_batch, x_future)      # [bs, time_dim, x_dim]

                # stl loss
                loss_stl = model.STL_loss(complete_traj, formula, formula_input_func, scale=stl_scale_value)
                loss_stl_true = model.STL_loss(complete_traj, formula, formula_input_func, scale=-1)

                # total loss
                loss = hps.weight_recon * loss_recon + weight_stl * loss_stl

                if (train_iteration % 10) == 0:
                    
                    torch.save({
                       'train_iteration': train_iteration,
                       'gradient_step': gradient_step,
                       'eval_iteration': eval_iteration,
                       'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'loss': loss,
                       'ic': ic},
                       save_model_path + "/model")

                nan_flag = False
                # take gradient steps with each loss term to see if they result in NaN after taking a gradient step.
                loss_stl.backward(retain_graph=True)
                if torch.stack([torch.isnan(p.grad).sum() for p in model.parameters()]).sum() > 0:
                    write_log(log_dir, "Training: Backpropagation through STL resulted in NaN. Saving data in nan folder")
                    print("Backpropagation through STL loss resulted in NaN")
                    torch.save({
                               'train_iteration': train_iteration,
                               'gradient_step': gradient_step,
                               'eval_iteration': eval_iteration,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'loss': loss,
                               'ic': ic},
                               '../nan/' + model_name + '/model_stl')
                    np.save('../nan/' + model_name + '/ic_stl.npy', ic.cpu())
                    nan_flag = True
                optimizer.zero_grad()

                loss_recon.backward(retain_graph=True)
                if torch.stack([torch.isnan(p.grad).sum() for p in model.parameters()]).sum() > 0:
                    write_log(log_dir, "Training: Backpropagation through recon resulted in NaN. Saving data in nan folder")
                    print("Backpropagation through recon loss resulted in NaN")
                    torch.save({
                               'train_iteration': train_iteration,
                               'gradient_step': gradient_step,
                               'eval_iteration': eval_iteration,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'loss': loss,
                               'ic': ic},
                               '../nan/' + model_name + '/model_recon')
                    np.save('../nan/' + model_name + '/ic_recon.npy', ic.cpu())
                    nan_flag = True
                optimizer.zero_grad()

                if writer is not None:
                    writer.add_scalar('train/loss/state', loss_state, gradient_step)
                    writer.add_scalar('train/loss/ctrl', loss_ctrl, gradient_step)
                    writer.add_scalar('train/loss/STL', loss_stl, gradient_step)
                    writer.add_scalar('train/loss/STL_true', loss_stl_true, gradient_step)
                    writer.add_scalar('train/loss/total', loss, gradient_step)
                    writer.add_scalar('train/parameters/teacher_training', teacher_training_value, gradient_step)
                    writer.add_scalar('train/parameters/stl_scale', stl_scale_value, gradient_step)
                    writer.add_scalar('train/parameters/weight_stl', weight_stl, gradient_step)

                if nan_flag:
                    # don't take step, wait for a new batch
                    continue
                loss.backward()
                optimizer.step()
                gradient_step += 1



            # plotting progress
            if (train_iteration % plot_freq) == 0:

                torch.save({
                            'train_iteration': train_iteration,
                            'gradient_step': gradient_step,
                            'eval_iteration': eval_iteration,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            'ic': ic},
                            save_model_path + "/model_{:02d}_iteration={:03d}".format(number, train_iteration))

                # compute trajectories with fixed imgs
                # with new ICs, propagate the trajectories
                ic_bs = ic.shape[0]
                xdim = ic.shape[-1]
                img_bs = env_param_ic_plotting.shape[0]

                ic_batch = ic[None,...].repeat([img_bs,1,1,1]).view(-1, 1, xdim)
                img_batch = ic_imgs_plotting.unsqueeze(1).repeat([1, ic_bs, 1, 1, 1]).view([-1, *ic_imgs_plotting.shape[-3:]])

                x_future, u_future = model.propagate_n(T, ic_batch, img_batch)
                complete_traj = model.join_partial_future_signal(ic_batch, x_future)      # [bs, time_dim, x_dim]

                # plotting trajectories from initial condition
                # trajectories from propagating initial states
                traj_np = model.unstandardize_x(complete_traj).view(img_bs, ic_bs, T+1, xdim).cpu().detach()

                # trajectory plot
                fig1, ax1s = plt.subplots(2, img_bs//2 , figsize=(20,10))
                for i in range(img_bs):
                    update_environment(case, env_tmp, env_param_ic_plotting[i], img_bs)
                    ax = ax1s[i//(img_bs//2), i % (img_bs//2)]
                    ax.grid(zorder=0)
                    _, ax = env_tmp.draw2D(ax=ax, kwargs=draw_params)
                    ax.axis("equal")
                    ax.set_title("%i: cover_x = %.1f"%(i+1, env_tmp.covers[0].center[0]))
                    ax.plot(traj_np[i,:,:,0].T, traj_np[i,:,:,1].T, alpha=0.4, c='RoyalBlue', zorder=5)
                    ax.scatter(traj_np[i,:,:,0].T, traj_np[i,:,:,1].T, alpha=0.4, c='RoyalBlue', zorder=5)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

                if writer is not None:
                    writer.add_figure('train/trajectories_from_ic', fig1, gradient_step)

                # trajectories from expert
                expert_bs = env_param_train.shape[0]
                expert_mini_bs = hps.expert_mini_bs

                # trajectory from propagating initial state of expert trajectory
                x_future, u_future = model.propagate_n(T, x_train[:,:1,:], imgs_train)
                x_traj_prop = model.join_partial_future_signal(x_train[:,:1,:], x_future)
                x_traj_prop = model.unstandardize_x(x_traj_prop).view([-1,expert_mini_bs, *x_traj_prop.shape[-2:]]).detach().cpu().numpy()
                # trajectory from teacher training, and used for reconstruction loss (what the training sees)
                x_traj_pred = model.unstandardize_x(x_traj_pred).view([-1,expert_mini_bs, *x_traj_pred.shape[-2:]]).detach().cpu().numpy()


                width = (expert_bs//expert_mini_bs)//height

                fig2, ax2s = plt.subplots(height, width, figsize=(25,10))

                tls = tls_train.view(-1, expert_mini_bs).int() - 1

                for i in range(width * height):

                    p = env_param_train[i * expert_mini_bs]
                    update_environment(case, env_tmp, p, 1, carlength=carlength)
                    ax = ax2s[i//width, i % width]
                    ax.grid(zorder=0)
                    _, ax = env_tmp.draw2D(ax=ax, kwargs=draw_params)
                    ax.axis("equal")
                    # ax.set_title("cover_x = %.1f"%(env_tmp.covers[0].center[0]))
                    ax.plot(x_traj_prop[i,:,:,0].T, x_traj_prop[i,:,:,1].T, alpha=0.4, c='dodgerblue', zorder=5)
                    ax.scatter(x_traj_prop[i,:,:,0].T, x_traj_prop[i,:,:,1].T, alpha=0.4, c='dodgerblue', zorder=5)
                    for j in range(expert_mini_bs):
                        ax.plot(x_traj_pred[i,j,:tls[i,j],0].T, x_traj_pred[i,j,:tls[i,j],1].T, alpha=0.4, c='IndianRed', zorder=5)
                        ax.scatter(x_traj_pred[i,j,:tls[i,j],0].T, x_traj_pred[i,j,:tls[i,j],1].T, alpha=0.4, c='IndianRed', zorder=5)
                        ax.plot(x_true_train[i,j,:tls[i,j],0].T, x_true_train[i,j,:tls[i,j],1].T, alpha=0.4, c='ForestGreen', zorder=2)
                        ax.scatter(x_true_train[i,j,:tls[i,j],0].T, x_true_train[i,j,:tls[i,j],1].T, alpha=0.4, c='ForestGreen', zorder=2)
                    ax.axis("equal")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)


                if writer is not None:
                    writer.add_figure('train/trajectories_from_expert', fig2, gradient_step)

                # plotting controls
                u_traj_prop = model.unstandardize_u(u_future).cpu().detach().view([-1,expert_mini_bs, *u_future.shape[-2:]])
                
                fig3, ax3s = plt.subplots(height, width, figsize=(20,8))
                ctrl_idx = 0
                for i in range(width * height):
                    ax = ax3s[i//width, i % width]
                    ax.plot(time, u_traj_prop[i,:,:,ctrl_idx].T, alpha=0.4, c='dodgerblue', zorder=5)
                    ax.scatter(time, u_traj_prop[i,:,:,ctrl_idx].T, alpha=0.4, c='dodgerblue', zorder=5)
                    for j in range(expert_mini_bs):
                        time_j = torch.arange(tls[i,j]) * model.dt
                        ax.plot(time_j, u_true_train[i,j,:tls[i,j],ctrl_idx].T, alpha=0.4, c='IndianRed', zorder=5)
                        ax.scatter(time_j, u_true_train[i,j,:tls[i,j],ctrl_idx].T, alpha=0.4, c='IndianRed', zorder=5)
                        ax.grid()
                        ax.set_xlim([0,T * model.dt])
                        ax.set_ylim([-3.5, 3.5])
                if writer is not None:
                    writer.add_figure('train/acceleration_from_expert', fig3, gradient_step)

                fig4, ax4s = plt.subplots(height, width, figsize=(20,8))
                ctrl_idx = 1
                for i in range(width * height):
                    ax = ax4s[i//width, i % width]
                    ax.plot(time, u_traj_prop[i,:,:,ctrl_idx].T, alpha=0.4, c='dodgerblue', zorder=5)
                    ax.scatter(time, u_traj_prop[i,:,:,ctrl_idx].T, alpha=0.4, c='dodgerblue', zorder=5)
                    for j in range(expert_mini_bs):
                        time_j = torch.arange(tls[i,j]) * model.dt
                        ax.plot(time_j, u_true_train[i,j,:tls[i,j],ctrl_idx].T, alpha=0.4, c='IndianRed', zorder=5)
                        ax.scatter(time_j, u_true_train[i,j,:tls[i,j],ctrl_idx].T, alpha=0.4, c='IndianRed', zorder=5)
                        ax.grid()
                        ax.set_xlim([0,T * model.dt])
                        ax.set_ylim([-0.4, 0.4])

                if writer is not None:
                    writer.add_figure('train/steering_from_expert', fig3, gradient_step)

                fig1.savefig(fig_dir + '/train/ic_trajectory_number={:02d}_iteration={:03d}.png'.format(number, train_iteration))
                fig2.savefig(fig_dir + '/train/expert_trajectory_number={:02d}_iteration={:03d}.png'.format(number, train_iteration))
                fig3.savefig(fig_dir + '/train/acceleration_number={:02d}_iteration={:03d}.png'.format(number, train_iteration))
                fig4.savefig(fig_dir + '/train/delta_number={:02d}_iteration={:03d}.png'.format(number, train_iteration))
                
                plt.close(fig1)
                plt.close(fig2)
                plt.close(fig3)
                plt.close(fig4)

            train_iteration += 1 



            # evaluation set
            model.eval()

            for (batch_idx, ic_) in enumerate(eval_loader):
                ic = ic_["ic"].to(device).float()        # [bs, 1, x_dim]
                model.eval()


                # reconstruct the expert model
                loss_state, loss_ctrl, x_traj_pred, u_traj_pred = model.reconstruction_loss(x_eval, u_eval, imgs_eval, tls_eval, teacher_training=1.0)
                loss_recon = loss_state + hps.weight_ctrl * loss_ctrl


                # with new ICs, propagate the trajectories
                ic_bs = ic.shape[0]
                xdim = ic.shape[-1]
                img_bs = hps.img_bs

                ps = sample_and_update_environment(case, model.env, img_bs, ic_bs)
                img_bs = ps.shape[0]
                
                ic_imgs = torch.stack([generate_img_tensor_from_parameter(case, pi) for pi in ps]).to(device)
                ic_batch = ic[None,...].repeat([img_bs,1,1,1]).view(-1, 1, xdim)
                img_batch = ic_imgs.unsqueeze(1).repeat([1, ic_bs, 1, 1, 1]).view([-1, *ic_imgs.shape[-3:]])

                x_future, u_future = model.propagate_n(T, ic_batch, img_batch)
                complete_traj = model.join_partial_future_signal(ic_batch, x_future)      # [bs, time_dim, x_dim]

                # stl loss
                loss_stl = model.STL_loss(complete_traj, formula, formula_input_func, scale=stl_scale_value)
                loss_stl_true = model.STL_loss(complete_traj, formula, formula_input_func, scale=-1)

                # total loss
                loss = hps.weight_recon * loss_recon + weight_stl * loss_stl

                if writer is not None:
                    writer.add_scalar('eval/state', loss_state, eval_iteration)
                    writer.add_scalar('eval/ctrl', loss_ctrl, eval_iteration)
                    writer.add_scalar('eval/STL', loss_stl, eval_iteration)
                    writer.add_scalar('eval/total', loss, eval_iteration)

                # plotting
                if (eval_iteration % plot_freq) == 0:

                    # compute trajectories with fixed imgs
                    # with new ICs, propagate the trajectories
                    img_bs = len(env_param_ic_plotting)
                    
                    ic_batch = ic[None,...].repeat([img_bs,1,1,1]).view(-1, 1, xdim)
                    img_batch = ic_imgs_plotting.unsqueeze(1).repeat([1, ic_bs, 1, 1, 1]).view([-1, *ic_imgs_plotting.shape[-3:]])


                    x_future, u_future = model.propagate_n(T, ic_batch, img_batch)
                    complete_traj = model.join_partial_future_signal(ic_batch, x_future)      # [bs, time_dim, x_dim]

                    # plotting trajectories from initial condition
                    # trajectories from propagating initial states
                    traj_np = model.unstandardize_x(complete_traj).view(img_bs, ic_bs, T+1, xdim).cpu().detach()

                    # trajectory plot
                    fig1, ax1s = plt.subplots(2, img_bs//2, figsize=(20,10))


                    for i in range(img_bs):
                        update_environment(case, env_tmp, env_param_ic_plotting[i], img_bs)
                        ax = ax1s[i//(img_bs//2), i % (img_bs//2)]
                        ax.grid(zorder=0)
                        _, ax = env_tmp.draw2D(ax=ax, kwargs=draw_params)
                        ax.axis("equal")
                        ax.plot(traj_np[i,:,:,0].T, traj_np[i,:,:,1].T, alpha=0.4, c='RoyalBlue', zorder=5)
                        ax.scatter(traj_np[i,:,:,0].T, traj_np[i,:,:,1].T, alpha=0.4, c='RoyalBlue', zorder=5)
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)
                    if writer is not None:
                        writer.add_figure('eval/trajectories_from_ic', fig1, gradient_step)

                    # trajectories from expert
                    expert_bs = env_param_eval.shape[0]
                    expert_mini_bs = hps.expert_mini_bs

                    # trajectory from propagating initial state of expert trajectory
                    x_future, u_future = model.propagate_n(T, x_eval[:,:1,:], imgs_eval)
                    x_traj_prop = model.join_partial_future_signal(x_eval[:,:1,:], x_future)
                    x_traj_prop = model.unstandardize_x(x_traj_prop).view([-1,expert_mini_bs, *x_traj_prop.shape[-2:]]).detach().cpu().numpy()
                    # trajectory from teacher training, and used for reconstruction loss (what the training sees)
                    x_traj_pred = model.unstandardize_x(x_traj_pred).view([-1,expert_mini_bs, *x_traj_pred.shape[-2:]]).detach().cpu().numpy()

                    width = (expert_bs//expert_mini_bs)//height

                    fig2, ax2s = plt.subplots(height, width, figsize=(25,10))
                    tls = tls_eval.view(-1, expert_mini_bs).int() - 1

                    for i in range(width * height):
                        p = env_param_eval[i * expert_mini_bs]
                        update_environment(case, env_tmp, p, 1, carlength=carlength)
                        ax = ax2s[i//width, i % width]
                        ax.grid(zorder=0)
                        _, ax = env_tmp.draw2D(ax=ax, kwargs=draw_params)
                        ax.axis("equal")
                        # ax.set_title("cover_x = %.1f"%(env_tmp.covers[0].center[0]))
                        ax.plot(x_traj_prop[i,:,:,0].T, x_traj_prop[i,:,:,1].T, alpha=0.4, c='dodgerblue', zorder=5)
                        ax.scatter(x_traj_prop[i,:,:,0].T, x_traj_prop[i,:,:,1].T, alpha=0.4, c='dodgerblue', zorder=5)
                        for j in range(expert_mini_bs):
                            ax.plot(x_traj_pred[i,j,:tls[i,j],0].T, x_traj_pred[i,j,:tls[i,j],1].T, alpha=0.4, c='IndianRed', zorder=5)
                            ax.scatter(x_traj_pred[i,j,:tls[i,j],0].T, x_traj_pred[i,j,:tls[i,j],1].T, alpha=0.4, c='IndianRed', zorder=5)
                            ax.plot(x_true_eval[i,j,:tls[i,j],0].T, x_true_eval[i,j,:tls[i,j],1].T, alpha=0.4, c='ForestGreen', zorder=2)
                            ax.scatter(x_true_eval[i,j,:tls[i,j],0].T, x_true_eval[i,j,:tls[i,j],1].T, alpha=0.4, c='ForestGreen', zorder=2)
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)

                    if writer is not None:
                        writer.add_figure('eval/trajectories_from_expert', fig2, gradient_step)

                    # plotting controls
                    u_traj_prop = model.unstandardize_u(u_future).cpu().detach().view([-1,expert_mini_bs, *u_future.shape[-2:]])

                    fig3, ax3s = plt.subplots(height, width, figsize=(20,8))
                    ctrl_idx = 0
                    for i in range(width * height):
                        ax = ax3s[i//width, i % width]
                        ax.plot(time, u_traj_prop[i,:,:,ctrl_idx].T, alpha=0.4, c='dodgerblue', zorder=5)
                        ax.scatter(time, u_traj_prop[i,:,:,ctrl_idx].T, alpha=0.4, c='dodgerblue', zorder=5)
                        for j in range(expert_mini_bs):
                            time_j = torch.arange(tls[i,j]) * model.dt
                            ax.plot(time_j, u_true_eval[i,j,:tls[i,j],ctrl_idx].T, alpha=0.4, c='IndianRed', zorder=5)
                            ax.scatter(time_j, u_true_eval[i,j,:tls[i,j],ctrl_idx].T, alpha=0.4, c='IndianRed', zorder=5)
                            ax.grid()
                            ax.set_xlim([0,T * model.dt])
                            ax.set_ylim([-3.5, 3.5])
                    if writer is not None:
                        writer.add_figure('eval/acceleration_from_expert', fig3, gradient_step)

                    fig4, ax4s = plt.subplots(height, width, figsize=(20,8))
                    ctrl_idx = 1
                    for i in range(width * height):
                        ax = ax4s[i//width, i % width]
                        ax.plot(time, u_traj_prop[i,:,:,ctrl_idx].T, alpha=0.4, c='dodgerblue', zorder=5)
                        ax.scatter(time, u_traj_prop[i,:,:,ctrl_idx].T, alpha=0.4, c='dodgerblue', zorder=5)
                        for j in range(expert_mini_bs):
                            time_j = torch.arange(tls[i,j]) * model.dt
                            ax.plot(time_j, u_true_eval[i,j,:tls[i,j],ctrl_idx].T, alpha=0.4, c='IndianRed', zorder=5)
                            ax.scatter(time_j, u_true_eval[i,j,:tls[i,j],ctrl_idx].T, alpha=0.4, c='IndianRed', zorder=5)
                            ax.grid()
                            ax.set_xlim([0,T * model.dt])
                            ax.set_ylim([-0.4, 0.4])

                    if writer is not None:
                        writer.add_figure('eval/steering_from_expert', fig3, gradient_step)

                    fig1.savefig(fig_dir + '/eval/ic_trajectory_number={:02d}_iteration={:03d}.png'.format(number, eval_iteration))
                    fig2.savefig(fig_dir + '/eval/expert_trajectory_number={:02d}_iteration={:03d}.png'.format(number, eval_iteration))
                    fig3.savefig(fig_dir + '/eval/acceleration_number={:02d}_iteration={:03d}.png'.format(number, eval_iteration))
                    fig4.savefig(fig_dir + '/eval/delta_number={:02d}_iteration={:03d}.png'.format(number, eval_iteration))
                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig3)
                    plt.close(fig4)
                break
                
            eval_iteration += 1
                



            # Feel free to modify the progress bar
            pbar.set_postfix(loss='{:.2e}'.format(loss))
            pbar.update(1)
            # Log summaries
            # training progress





# def train(model, train_traj, eval_traj, formula, formula_input_func, train_loader, eval_loader, device, tqdm, writer, hps, save_model_path, number, iter_max=np.inf, status="new", xlim=[-6, 16], ylim=[-6, 16]):
#     '''
#     This controls the control logic of the gradient steps

#     Inputs:
#     model              : The neural network model (refer to learning.py)               
#     train_traj         : Expert trajectory training set             
#     eval_traj          : Expert trajectory evaluation set         
#     formula            : STL specification to be satisfied  
#     formula_input_func : STL input function corresponding to STL specification
#     train_loader       : PyTorch data loader for initial condition train set
#     eval_loader        : PyTorch data loader for initial condition eval set
#     device             : cpu or cuda
#     tqdm               : progress bar object
#     writer             : Tensorboard writer object
#     hps                : hyperparameters for training (refer to run.py)
#     save_model_path    : path where the model is saved
#     number             : iteration for the training-adversarial loop
#     iter_max           : maximum number of training epochs
#     '''

#     log_dir = save_model_path.replace("models", "logs", 1)
#     fig_dir = save_model_path.replace("models", "figs", 1)

#     optimizer = torch.optim.Adam(model.parameters(), weight_decay=hps.weight_decay)
#     model_name = save_model_path.split("/")[-1]

#     # if training from scratch, initialize training steps at 0
#     if status == "new":
#         train_iteration = 0 # train iteration number
#         eval_iteration = 0 # evaluation iteration number
#         gradient_step = 0 # descent number
#     # if continuing training the model, load the last iteration values
#     elif status == "continue":
#         print("\nContinue training model:", save_model_path, "\n")
#         checkpoint = torch.load(save_model_path + "/model")
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         train_iteration =  checkpoint['train_iteration']
#         gradient_step =  checkpoint['gradient_step']
#         eval_iteration =  checkpoint['eval_iteration']

#     # switch everything to the specified device
#     x_train, u_train = train_traj[0].to(device), train_traj[1].to(device)
#     x_eval, u_eval = eval_traj[0].to(device), eval_traj[1].to(device)
#     model = model.to(device)
#     model.switch_device(device)

#     x_train_ = model.unstandardize_x(x_train)
#     u_train_ = model.unstandardize_u(u_train)
#     x_eval_ = model.unstandardize_x(x_eval)
#     u_eval_ = model.unstandardize_u(u_eval)

#     T = x_train.shape[1] + 10

#     with tqdm(total=(iter_max - train_iteration)) as pbar:
#         while True:
#             if train_iteration == iter_max:
#                 torch.save({
#                            'train_iteration': train_iteration,
#                            'gradient_step': gradient_step,
#                            'eval_iteration': eval_iteration,
#                            'model_state_dict': model.state_dict(),
#                            'optimizer_state_dict': optimizer.state_dict(),
#                            'loss': loss,
#                            'ic': ic},
#                            save_model_path + "/model")

#                 torch.save({
#                            'train_iteration': train_iteration,
#                            'gradient_step': gradient_step,
#                            'eval_iteration': eval_iteration,
#                            'model_state_dict': model.state_dict(),
#                            'optimizer_state_dict': optimizer.state_dict(),
#                            'loss': loss,
#                            'ic': ic},
#                            save_model_path + "/model_{:02d}".format(number))
#                 return

#             hps_idx = train_iteration % (iter_max // (number + 1))
#             for (batch_idx, ic_) in enumerate(train_loader):
                
#                 optimizer.zero_grad()
#                 ic = ic_.to(device).float()        # [bs, 1, x_dim]
#                 model.train()
#                 # parameters
#                 teacher_training_value = hps.teacher_training(hps_idx)
#                 weight_stl = hps.weight_stl(hps_idx)
#                 stl_scale_value = hps.stl_scale(hps_idx)

#                 # reconstruct the expert model
#                 loss_state, loss_ctrl, x_traj_pred, u_traj_pred = model.reconstruction_loss(x_train, u_train, teacher_training=teacher_training_value)
#                 loss_recon = loss_state + hps.weight_ctrl * loss_ctrl

#                 # with new ICs, propagate the trajectories
#                 x_future, u_future = model.propagate_n(T, ic)
#                 complete_traj = model.join_partial_future_signal(ic, x_future)      # [bs, time_dim, x_dim]

#                 # stl loss
#                 loss_stl = model.STL_loss(complete_traj, formula, formula_input_func, scale=stl_scale_value)
#                 loss_stl_true = model.STL_loss(complete_traj, formula, formula_input_func, scale=-1)

#                 # total loss
#                 loss = hps.weight_recon * loss_recon + weight_stl * loss_stl

#                 if (train_iteration % 10) == 0:
                    
#                     torch.save({
#                        'train_iteration': train_iteration,
#                        'gradient_step': gradient_step,
#                        'eval_iteration': eval_iteration,
#                        'model_state_dict': model.state_dict(),
#                        'optimizer_state_dict': optimizer.state_dict(),
#                        'loss': loss,
#                        'ic': ic},
#                        save_model_path + "/model")


#                 nan_flag = False
#                 # take gradient steps with each loss term to see if they result in NaN after taking a gradient step.
#                 loss_stl.backward(retain_graph=True)
#                 if torch.stack([torch.isnan(p.grad).sum() for p in model.parameters()]).sum() > 0:
#                     write_log(log_dir, "Training: Backpropagation through STL resulted in NaN. Saving data in nan folder")
#                     print("Backpropagation through STL loss resulted in NaN")
#                     torch.save({
#                                'train_iteration': train_iteration,
#                                'gradient_step': gradient_step,
#                                'eval_iteration': eval_iteration,
#                                'model_state_dict': model.state_dict(),
#                                'optimizer_state_dict': optimizer.state_dict(),
#                                'loss': loss,
#                                'ic': ic},
#                                '../nan/' + model_name + '/model_stl')
#                     np.save('../nan/' + model_name + '/ic_stl.npy', ic.cpu())
#                     nan_flag = True
#                 optimizer.zero_grad()

#                 loss_recon.backward(retain_graph=True)
#                 if torch.stack([torch.isnan(p.grad).sum() for p in model.parameters()]).sum() > 0:
#                     write_log(log_dir, "Training: Backpropagation through recon resulted in NaN. Saving data in nan folder")
#                     print("Backpropagation through recon loss resulted in NaN")
#                     torch.save({
#                                'train_iteration': train_iteration,
#                                'gradient_step': gradient_step,
#                                'eval_iteration': eval_iteration,
#                                'model_state_dict': model.state_dict(),
#                                'optimizer_state_dict': optimizer.state_dict(),
#                                'loss': loss,
#                                'ic': ic},
#                                '../nan/' + model_name + '/model_recon')
#                     np.save('../nan/' + model_name + '/ic_recon.npy', ic.cpu())
#                     nan_flag = True
#                 optimizer.zero_grad()
#                 if writer is not None:
#                     writer.add_scalar('train/loss/state', loss_state, gradient_step)
#                     writer.add_scalar('train/loss/ctrl', loss_ctrl, gradient_step)
#                     writer.add_scalar('train/loss/STL', loss_stl, gradient_step)
#                     writer.add_scalar('train/loss/STL_true', loss_stl_true, gradient_step)
#                     writer.add_scalar('train/loss/total', loss, gradient_step)
#                     writer.add_scalar('train/parameters/teacher_training', teacher_training_value, gradient_step)
#                     writer.add_scalar('train/parameters/stl_scale', stl_scale_value, gradient_step)
#                     writer.add_scalar('train/parameters/weight_stl', weight_stl, gradient_step)

#                 # plotting progress
#                 if batch_idx % 20 == 0:
#                     # trajectories from propagating initial states
#                     traj_np = model.unstandardize_x(complete_traj).cpu().detach().numpy()
#                     # trajectory from propagating initial state of expert trajectory
#                     x_future, u_future = model.propagate_n(T, x_train[:,:1,:])
#                     x_traj_prop = model.join_partial_future_signal(x_train[:,:1,:], x_future)
#                     x_traj_prop = model.unstandardize_x(x_traj_prop).squeeze().detach().cpu().numpy()
#                     # trajectory from teacher training, and used for reconstruction loss (what the training sees)
#                     x_traj_pred = model.unstandardize_x(x_traj_pred)

#                     # trajectory plot
#                     fig1, ax = plt.subplots(figsize=(10,10))
#                     _, ax = model.env.draw2D(ax=ax, kwargs=draw_params)
                    

#                     # plotting the sampled initial state trajectories
#                     ax.plot(traj_np[:,:,0].T, traj_np[:,:,1].T, alpha=0.4, c='RoyalBlue')
#                     ax.scatter(traj_np[:,:,0].T, traj_np[:,:,1].T, alpha=0.4, c='RoyalBlue')
#                     # plotting true expert trajectory
#                     ax.plot(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], linewidth=4, c='k', linestyle='--', zorder=2)
#                     ax.scatter(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], s=100, c='k', label="Expert", zorder=3)
#                     # plotting propagated expert trajectory
#                     ax.plot(x_traj_prop[:,0], x_traj_prop[:,1], linewidth=3, c="dodgerblue", linestyle='--', label="Reconstruction", zorder=5)
#                     ax.scatter(x_traj_prop[:,0], x_traj_prop[:,1], s=100, c="dodgerblue", zorder=5)
#                     # plotting predicted expert trajectory during training (with teacher training)
#                     ax.plot(x_traj_pred.cpu().detach().squeeze().numpy()[:,0], x_traj_pred.cpu().detach().squeeze().numpy()[:,1], linewidth=3, c="IndianRed", linestyle='--', label="Expert recon.", zorder=6)
#                     ax.scatter(x_traj_pred.cpu().detach().squeeze().numpy()[:,0], x_traj_pred.cpu().detach().squeeze().numpy()[:,1], s=100, c="IndianRed", zorder=7)
#                     ax.axis("equal")
#                     ax.set_xlim(xlim)
#                     ax.set_ylim(ylim)

#                     if writer is not None:
#                         writer.add_figure('train/trajectory', fig1, gradient_step)

#                     # controls plot
#                     fig2, axs = plt.subplots(1,2,figsize=(15,6))
#                     for (k,a) in enumerate(axs):
#                         a.plot(u_train_.squeeze().cpu().detach().numpy()[:,k], label="Expert")
#                         a.plot(model.unstandardize_u(u_future).squeeze().cpu().detach().numpy()[:,k], linestyle='--', label="Reconstruction")
#                         a.grid()
#                         a.set_xlim([0,T])
#                         a.set_ylim([-4,4])

#                     if writer is not None:
#                         writer.add_figure('train/controls', fig2, gradient_step)


#                 if nan_flag:
#                     # don't take step, wait for a new batch
#                     continue
#                 loss.backward()
#                 optimizer.step()
#                 gradient_step += 1

#             fig1.savefig(fig_dir + '/train/number={:02d}_iteration={:03d}.png'.format(number, train_iteration))

#             torch.save({
#                         'train_iteration': train_iteration,
#                         'gradient_step': gradient_step,
#                         'eval_iteration': eval_iteration,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'loss': loss,
#                         'ic': ic},
#                         save_model_path + "/model_{:02d}_iteration={:03d}.png".format(number, train_iteration))

#             # evaluation set
#             model.eval()
#             for (batch_idx, ic_) in enumerate(eval_loader):
              
#                 ic = ic_.to(device).float()        # [bs, 1, x_dim]
#                 model.eval()


#                 # reconstruct the expert model
#                 loss_state, loss_ctrl, x_traj_pred, u_traj_pred = model.reconstruction_loss(x_eval, u_eval, teacher_training=1.0)
#                 loss_recon = loss_state + hps.weight_ctrl * loss_ctrl

#                 # with new ICs, propagate the trajectories
#                 x_future, u_future = model.propagate_n(T, ic)
#                 complete_traj = model.join_partial_future_signal(ic, x_future)      # [time, bs, x_dim]

#                 # stl loss
#                 loss_stl = model.STL_loss(complete_traj, formula, formula_input_func, scale=-1)

#                 # total loss
#                 loss = hps.weight_recon * loss_recon + weight_stl * loss_stl

#                 if writer is not None:
#                     writer.add_scalar('eval/state', loss_state, eval_iteration)
#                     writer.add_scalar('eval/ctrl', loss_ctrl, eval_iteration)
#                     writer.add_scalar('eval/STL', loss_stl, eval_iteration)
#                     writer.add_scalar('eval/total', loss, eval_iteration)


#                 traj_np = model.unstandardize_x(complete_traj).cpu().detach().numpy()
#                 # trajectory from propagating initial state of expert trajectory
#                 x_future, u_future = model.propagate_n(T, x_eval[:,:1,:])
#                 x_traj_prop = model.join_partial_future_signal(x_eval[:,:1,:], x_future)
#                 x_traj_prop = model.unstandardize_x(x_traj_prop).squeeze().detach().cpu().numpy()
#                 # trajectory from teacher training, and used for reconstruction loss (what the training sees)
#                 x_traj_pred = model.unstandardize_x(x_traj_pred)

#                 fig1, ax = plt.subplots(figsize=(10,10))
#                 _, ax = model.env.draw2D(ax=ax, kwargs=draw_params)
#                 ax.axis("equal")

#                 # plotting the sampled initial state trajectories
#                 ax.plot(traj_np[:,:,0].T, traj_np[:,:,1].T, alpha=0.4, c='RoyalBlue')
#                 ax.scatter(traj_np[:,:,0].T, traj_np[:,:,1].T, alpha=0.4, c='RoyalBlue')
#                  # plotting true expert trajectory
#                 ax.plot(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], linewidth=4, c='k', linestyle='--', zorder=2)
#                 ax.scatter(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], s=100, c='k', label="Expert", zorder=3)
#                 # plotting propagated expert trajectory
#                 ax.plot(x_traj_prop[:,0], x_traj_prop[:,1], linewidth=3, c="dodgerblue", linestyle='--', label="Reconstruction", zorder=5)
#                 ax.scatter(x_traj_prop[:,0], x_traj_prop[:,1], s=100, c="dodgerblue", zorder=5)
#                 # plotting predicted expert trajectory during training (with teacher training)
#                 ax.plot(x_traj_pred.cpu().detach().squeeze().numpy()[:,0], x_traj_pred.cpu().detach().squeeze().numpy()[:,1], linewidth=3, c="IndianRed", linestyle='--', label="Expert recon.", zorder=6)
#                 ax.scatter(x_traj_pred.cpu().detach().squeeze().numpy()[:,0], x_traj_pred.cpu().detach().squeeze().numpy()[:,1], s=100, c="IndianRed", zorder=7)

#                 ax.set_xlim(xlim)
#                 ax.set_ylim(ylim)
#                 if writer is not None:
#                     writer.add_figure('eval/trajectory', fig1, eval_iteration)
#                 fig1.savefig(fig_dir + '/eval/number={:02d}_iteration={:03d}.png'.format(number, eval_iteration))

#                 fig2, axs = plt.subplots(1,2,figsize=(15,6))
#                 for (k,a) in enumerate(axs):
#                     a.plot(u_train_.squeeze().cpu().detach().numpy()[:,k], label="Expert")
#                     a.plot(model.unstandardize_u(u_future).squeeze().cpu().detach().numpy()[:,k], linestyle='--', label="Reconstruction")
#                     a.grid()
#                     a.set_xlim([0,T])
#                     a.set_ylim([-4,4])
#                 if writer is not None:
#                     writer.add_figure('eval/controls', fig2, eval_iteration)


#                 eval_iteration += 1
#                 break

#             train_iteration += 1 # i is num of runs through the data set



#             # Feel free to modify the progress bar
#             pbar.set_postfix(loss='{:.2e}'.format(loss))
#             pbar.update(1)
#             # Log summaries
#             # training progress