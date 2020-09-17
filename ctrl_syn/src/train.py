import sys
import torch
import os
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
import IPython
from torch.utils.data import Dataset, DataLoader
from learning import *
from utils import *

draw_params = {"initial": {"color": "lightskyblue", "fill": True, "alpha": 0.5}, "final": {"color": "coral", "fill": True, "alpha": 0.5}, "covers": {"color": "black", "fill": False}, "obs": {"color": "red", "fill": True, "alpha": 0.5} }


def train(model, train_traj, eval_traj, formula, formula_input_func, train_loader, eval_loader, device, tqdm, writer, hps, save_model_path, number, iter_max=np.inf, status="new"):
    log_dir = save_model_path.replace("models", "logs", 1)
    fig_dir = save_model_path.replace("models", "figs", 1)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=hps.weight_decay)
    model_name = save_model_path.split("/")[-1]
    if status == "new":
        train_iteration = 0 # train iteration number
        eval_iteration = 0 # evaluation iteration number
        gradient_step = 0 # descent number
    elif status == "continue":
        print("\nContinue training model:", save_model_path, "\n")
        checkpoint = torch.load(save_model_path + "/model")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_iteration =  checkpoint['train_iteration']
        gradient_step =  checkpoint['gradient_step']
        eval_iteration =  checkpoint['eval_iteration']


    x_train, u_train = train_traj[0].to(device), train_traj[1].to(device)
    x_eval, u_eval = eval_traj[0].to(device), eval_traj[1].to(device)
    model = model.to(device)
    model.switch_device(device)
    mu_x, sigma_x = model.stats[0][:,:,:4], model.stats[1][:,:,:4]
    mu_u, sigma_u = model.stats[0][:,:,4:], model.stats[1][:,:,4:]
    x_train_ = unstandardize_data(x_train, mu_x, sigma_x)
    u_train_ = unstandardize_data(u_train, mu_u, sigma_u)
    x_eval_ = unstandardize_data(x_eval, mu_x, sigma_x)
    u_eval_ = unstandardize_data(u_eval, mu_u, sigma_u)
    T = x_train.shape[0]+4
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
            for (batch_idx, ic_) in enumerate(train_loader):
                
                optimizer.zero_grad()
                ic = torch.cat([x_train[:1,:,:], ic_.permute([1,0,2]).to(device).float()], dim=1)    # [1, bs, x_dim]
                model.train()
                o, u, x_pred = model(x_train)

                # reconstruct the expert model
                # teacher_training_value = hps.teacher_training(train_iteration % (iter_max // (number + 1)))
                teacher_training_value = hps.teacher_training(train_iteration)

                loss_state, loss_ctrl = model.state_control_loss(x_train, x_train, u_train, teacher_training=teacher_training_value)
                loss_recon = loss_state + hps.weight_ctrl * loss_ctrl
                # with new ICs, propagate the trajectories and keep them inside the reachable set
                x_future, u_future = model.propagate_n(T, ic)
                complete_traj = model.join_partial_future_signal(ic, x_future)
                loss_HJI = model.HJI_loss(complete_traj)
                
                # stl loss
                stl_scale_value = hps.stl_scale(train_iteration % (iter_max // (number + 1)))
                # breakpoint()
                loss_stl = model.STL_loss(complete_traj, formula, formula_input_func, scale=stl_scale_value)
                loss_stl_true = model.STL_loss(complete_traj, formula, formula_input_func, scale=-1)
                loss_spc = model.safety_preserving_loss(complete_traj[:-1,:,:], u_future)

                # total loss

                weight_hji = hps.weight_hji(train_iteration % (iter_max // (number + 1)))
                weight_stl = hps.weight_stl(train_iteration % (iter_max // (number + 1)))
                loss = hps.weight_recon * loss_recon + weight_hji * (loss_HJI + 0.5 * loss_spc)  + weight_stl * loss_stl

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
                    np.save('../nan/' + model_name + '/ic_stl.npy', ic)
                    nan_flag = True
                optimizer.zero_grad()

                loss_HJI.backward(retain_graph=True)
                if torch.stack([torch.isnan(p.grad).sum() for p in model.parameters()]).sum() > 0:
                    write_log(log_dir, "Training: Backpropagation through HJI resulted in NaN. Saving data in nan folder")
                    print("Backpropagation through HJI loss resulted in NaN")
                    torch.save({
                               'train_iteration': train_iteration,
                               'gradient_step': gradient_step,
                               'eval_iteration': eval_iteration,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'loss': loss,
                               'ic': ic},
                               '../nan/' + model_name + '/model_hji')
                    np.save('../nan/' + model_name + '/ic_hji.npy', ic)
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
                    np.save('../nan/' + model_name + '/ic_recon.npy', ic)
                    nan_flag = True
                optimizer.zero_grad()


                loss_spc.backward(retain_graph=True)
                if torch.stack([torch.isnan(p.grad).sum() for p in model.parameters()]).sum() > 0:
                    write_log(log_dir, "Training: Backpropagation through SPC resulted in NaN. Saving data in nan folder")
                    print("Backpropagation through SPC loss resulted in NaN")
                    torch.save({
                               'train_iteration': train_iteration,
                               'gradient_step': gradient_step,
                               'eval_iteration': eval_iteration,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'loss': loss,
                               'ic': ic},
                               '../nan/' + model_name + '/model_stl')
                    np.save('../nan/' + model_name + '/ic_stl.npy', ic)
                    nan_flag = True
                optimizer.zero_grad()

                writer.add_scalar('train/loss/state', loss_state, gradient_step)
                writer.add_scalar('train/loss/ctrl', loss_ctrl, gradient_step)
                writer.add_scalar('train/loss/HJI', loss_HJI, gradient_step)
                writer.add_scalar('train/loss/STL', loss_stl, gradient_step)
                writer.add_scalar('train/loss/STL_true', loss_stl_true, gradient_step)
                writer.add_scalar('train/loss/SPC', loss_spc, gradient_step)

                writer.add_scalar('train/loss/total', loss, gradient_step)
                writer.add_scalar('train/parameters/teacher_training', teacher_training_value, gradient_step)
                writer.add_scalar('train/parameters/stl_scale', stl_scale_value, gradient_step)
                writer.add_scalar('train/parameters/weight_hji', weight_hji, gradient_step)
                writer.add_scalar('train/parameters/weight_stl', weight_stl, gradient_step)


                if batch_idx % 20 == 0:
                    traj_np = unstandardize_data(complete_traj, mu_x, sigma_x).cpu().detach().numpy()
                    x_future, u_future = model.propagate_n(T, x_train[:1,:,:])
                    p = model.join_partial_future_signal(x_train[:1,:,:], x_future)
                    p = unstandardize_data(p, mu_x, sigma_x).squeeze().detach().cpu().numpy()
                    x_pred = unstandardize_data(x_pred, mu_x, sigma_x)
                    fig1, ax = plt.subplots(figsize=(10,10))
                    _, ax = model.env.draw2D(ax=ax, kwargs=draw_params)
                    ax.axis("equal")
                    _, ax = plot_hji_contour(ax)
                    ax.plot(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], linewidth=4)
                    ax.scatter(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], s=100)
                    ax.plot(p[:,0], p[:,1], linewidth=4)
                    ax.scatter(p[:,0], p[:,1], marker='^', s=100)
                    ax.plot(x_pred.cpu().detach().squeeze().numpy()[:,0], x_pred.cpu().detach().squeeze().numpy()[:,1])
                    ax.scatter(x_pred.cpu().detach().squeeze().numpy()[:,0], x_pred.cpu().detach().squeeze().numpy()[:,1], marker='*', s=100)
                    for j in range(traj_np.shape[1]):
                        ax.plot(traj_np[:,j,0], traj_np[:,j,1])
                        ax.scatter(traj_np[:,j,0], traj_np[:,j,1])

                    ax.set_xlim([-5, 15])
                    ax.set_ylim([-2, 12])
                    writer.add_figure('train/trajectory', fig1, gradient_step)

                    fig2, axs = plt.subplots(1,2,figsize=(15,6))
                    for (k,a) in enumerate(axs):
                        a.plot(u_train_.squeeze().cpu().detach().numpy()[:,k])
                        a.plot(unstandardize_data(u_future, mu_u, sigma_u).squeeze().cpu().detach().numpy()[:,k],'--')
                        a.grid()
                        a.set_xlim([0, T])
                        a.set_ylim([-4,4])
                    writer.add_figure('train/controls', fig2, gradient_step)


                if nan_flag:
                    # don't take step, wait for a new batch
                    continue
                loss.backward()
                optimizer.step()
                gradient_step += 1

            fig1.savefig(fig_dir + '/train/number={:02d}_iteration={:03d}.png'.format(number, train_iteration))


            # evaluation set
            model.eval()
            for (batch_idx, ic_) in enumerate(eval_loader):
                ic = torch.cat([x_eval[:1,:,:], ic_.permute([1,0,2]).to(device).float()], dim=1)
                o, u, x_pred = model(x_eval)
                loss_state, loss_ctrl = model.state_control_loss(x_eval, x_eval, u_eval, teacher_training=1.0)
                loss_recon = loss_state + hps.weight_ctrl * loss_ctrl
                x_future, u_future = model.propagate_n(T, ic)
                complete_traj = model.join_partial_future_signal(ic, x_future)
                loss_HJI = model.HJI_loss(complete_traj)
                loss_stl = model.STL_loss(complete_traj, formula, formula_input_func, scale=-1)
                weight_hji = hps.weight_hji(train_iteration)
                weight_stl = hps.weight_stl(train_iteration)
                loss_spc = model.safety_preserving_loss(complete_traj[:-1,:,:], u_future)

                loss = hps.weight_recon * loss_recon + weight_hji * (loss_HJI + 0.5 * loss_spc) + weight_stl * loss_stl
                
                traj_np = unstandardize_data(complete_traj, mu_x, sigma_x).cpu().detach().numpy()
                x_future, u_future = model.propagate_n(T, x_eval[:1,:,:])
                p = model.join_partial_future_signal(x_eval[:1,:,:], x_future)
                p = unstandardize_data(p, mu_x, sigma_x).squeeze().detach().cpu().numpy()
                
                writer.add_scalar('eval/state', loss_state, eval_iteration)
                writer.add_scalar('eval/ctrl', loss_ctrl, eval_iteration)
                writer.add_scalar('eval/HJI', loss_HJI, eval_iteration)
                writer.add_scalar('eval/SPC', loss_spc, eval_iteration)
                writer.add_scalar('eval/STL', loss_stl, eval_iteration)
                writer.add_scalar('eval/total', loss, eval_iteration)
                x_pred = unstandardize_data(x_pred, mu_x, sigma_x)

                fig1, ax = plt.subplots(figsize=(10,10))
                _, ax = model.env.draw2D(ax=ax, kwargs=draw_params)
                ax.axis("equal")
                _, ax = plot_hji_contour(ax)
                ax.plot(x_eval_.squeeze().cpu().numpy()[:,0], x_eval_.squeeze().cpu().numpy()[:,1], linewidth=4)
                ax.scatter(x_eval_.squeeze().cpu().numpy()[:,0], x_eval_.squeeze().cpu().numpy()[:,1], s=100)
                ax.plot(p[:,0], p[:,1], linewidth=4)
                ax.scatter(p[:,0], p[:,1], marker='^', s=100)
                ax.plot(x_pred.squeeze().cpu().detach().numpy()[:,0], x_pred.squeeze().cpu().detach().numpy()[:,1])
                ax.scatter(x_pred.squeeze().cpu().detach().numpy()[:,0], x_pred.squeeze().cpu().detach().numpy()[:,1], marker='*', s=100)
                for j in range(traj_np.shape[1]):
                    ax.plot(traj_np[:,j,0], traj_np[:,j,1])
                    ax.scatter(traj_np[:,j,0], traj_np[:,j,1])

                ax.set_xlim([-5, 15])
                ax.set_ylim([-2, 12])
                writer.add_figure('eval/trajectory', fig1, eval_iteration)
                fig1.savefig(fig_dir + '/eval/number={:02d}_iteration={:03d}.png'.format(number, eval_iteration))

                
                fig2, axs = plt.subplots(1,2,figsize=(15,6))
                for (k,a) in enumerate(axs):
                    a.plot(u_train_.squeeze().cpu().detach().numpy()[:,k])
                    a.plot(unstandardize_data(u_future, mu_u, sigma_u).squeeze().cpu().detach().numpy()[:,k],'--')
                    a.grid()
                    a.set_xlim([0, T])
                    a.set_ylim([-4,4])
                writer.add_figure('eval/controls', fig2, eval_iteration)

                eval_iteration += 1
                break

            train_iteration += 1 # i is num of runs through the data set



            # Feel free to modify the progress bar
            pbar.set_postfix(loss='{:.2e}'.format(loss))
            pbar.update(1)
            # Log summaries
            # training progress

