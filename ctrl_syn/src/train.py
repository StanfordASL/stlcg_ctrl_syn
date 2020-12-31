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
    '''
    This controls the control logic of the gradient steps

    Inputs:
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
    model = model.to(device)
    model.switch_device(device)

    x_train_ = model.unstandardize_x(x_train)
    u_train_ = model.unstandardize_u(u_train)
    x_eval_ = model.unstandardize_x(x_eval)
    u_eval_ = model.unstandardize_u(u_eval)

    T = x_train.shape[1] + 10

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
                ic = ic_.to(device).float()        # [bs, 1, x_dim]
                model.train()
                # parameters
                teacher_training_value = hps.teacher_training(hps_idx)
                weight_stl = hps.weight_stl(hps_idx)
                stl_scale_value = hps.stl_scale(hps_idx)

                # reconstruct the expert model
                loss_state, loss_ctrl, x_traj_pred, u_traj_pred = model.reconstruction_loss(x_train, u_train, teacher_training=teacher_training_value)
                loss_recon = loss_state + hps.weight_ctrl * loss_ctrl

                # with new ICs, propagate the trajectories
                x_future, u_future = model.propagate_n(T, ic)
                complete_traj = model.join_partial_future_signal(ic, x_future)      # [bs, time_dim, x_dim]

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

                writer.add_scalar('train/loss/state', loss_state, gradient_step)
                writer.add_scalar('train/loss/ctrl', loss_ctrl, gradient_step)
                writer.add_scalar('train/loss/STL', loss_stl, gradient_step)
                writer.add_scalar('train/loss/STL_true', loss_stl_true, gradient_step)
                writer.add_scalar('train/loss/total', loss, gradient_step)
                writer.add_scalar('train/parameters/teacher_training', teacher_training_value, gradient_step)
                writer.add_scalar('train/parameters/stl_scale', stl_scale_value, gradient_step)
                writer.add_scalar('train/parameters/weight_stl', weight_stl, gradient_step)

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
                    ax.set_ylim([-5, 15])

                    writer.add_figure('train/trajectory', fig1, gradient_step)

                    # controls plot
                    fig2, axs = plt.subplots(1,2,figsize=(15,6))
                    for (k,a) in enumerate(axs):
                        a.plot(u_train_.squeeze().cpu().detach().numpy()[:,k], label="Expert")
                        a.plot(model.unstandardize_u(u_future).squeeze().cpu().detach().numpy()[:,k], linestyle='--', label="Reconstruction")
                        a.grid()
                        a.set_xlim([0,T])
                        a.set_ylim([-4,4])
                    writer.add_figure('train/controls', fig2, gradient_step)


                if nan_flag:
                    # don't take step, wait for a new batch
                    continue
                loss.backward()
                optimizer.step()
                gradient_step += 1

            fig1.savefig(fig_dir + '/train/number={:02d}_iteration={:03d}.png'.format(number, train_iteration))

            torch.save({
                        'train_iteration': train_iteration,
                        'gradient_step': gradient_step,
                        'eval_iteration': eval_iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'ic': ic},
                        save_model_path + "/model_{:02d}_iteration={:03d}.png".format(number, train_iteration))

            # evaluation set
            model.eval()
            for (batch_idx, ic_) in enumerate(eval_loader):
              
                ic = ic_.to(device).float()        # [bs, 1, x_dim]
                model.eval()


                # reconstruct the expert model
                loss_state, loss_ctrl, x_traj_pred, u_traj_pred = model.reconstruction_loss(x_eval, u_eval, teacher_training=1.0)
                loss_recon = loss_state + hps.weight_ctrl * loss_ctrl

                # with new ICs, propagate the trajectories
                x_future, u_future = model.propagate_n(T, ic)
                complete_traj = model.join_partial_future_signal(ic, x_future)      # [time, bs, x_dim]

                # stl loss
                loss_stl = model.STL_loss(complete_traj, formula, formula_input_func, scale=-1)

                # total loss
                loss = hps.weight_recon * loss_recon + weight_stl * loss_stl

                writer.add_scalar('eval/state', loss_state, eval_iteration)
                writer.add_scalar('eval/ctrl', loss_ctrl, eval_iteration)
                writer.add_scalar('eval/STL', loss_stl, eval_iteration)
                writer.add_scalar('eval/total', loss, eval_iteration)


                traj_np = model.unstandardize_x(complete_traj).cpu().detach().numpy()
                # trajectory from propagating initial state of expert trajectory
                x_future, u_future = model.propagate_n(T, x_eval[:,:1,:])
                x_traj_prop = model.join_partial_future_signal(x_eval[:,:1,:], x_future)
                x_traj_prop = model.unstandardize_x(x_traj_prop).squeeze().detach().cpu().numpy()
                # trajectory from teacher training, and used for reconstruction loss (what the training sees)
                x_traj_pred = model.unstandardize_x(x_traj_pred)

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
                ax.set_ylim([-5, 15])

                writer.add_figure('eval/trajectory', fig1, eval_iteration)
                fig1.savefig(fig_dir + '/eval/number={:02d}_iteration={:03d}.png'.format(number, eval_iteration))

                fig2, axs = plt.subplots(1,2,figsize=(15,6))
                for (k,a) in enumerate(axs):
                    a.plot(u_train_.squeeze().cpu().detach().numpy()[:,k], label="Expert")
                    a.plot(model.unstandardize_u(u_future).squeeze().cpu().detach().numpy()[:,k], linestyle='--', label="Reconstruction")
                    a.grid()
                    a.set_xlim([0,T])
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



def train_cnn(model, train_traj, eval_traj, imgs, tls, formula, formula_input_func, train_loader, eval_loader, device, tqdm, writer, hps, save_model_path, number, iter_max=np.inf, status="new"):
    '''
    This controls the control logic of the gradient steps

    Inputs:
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

    T = x_train.shape[1] + 10

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
                ic = ic_.to(device).float()        # [bs, 1, x_dim]
                model.train()
                # parameters
                teacher_training_value = hps.teacher_training(hps_idx)
                weight_stl = hps.weight_stl(hps_idx)
                stl_scale_value = hps.stl_scale(hps_idx)

                # reconstruct the expert model
                loss_state, loss_ctrl, x_traj_pred, u_traj_pred = model.reconstruction_loss(x_train, u_train, imgs_train, tls_train, teacher_training=teacher_training_value)
                loss_recon = loss_state + hps.weight_ctrl * loss_ctrl

                # with new ICs, propagate the trajectories
                bs = ic.shape[0]
                centers = np.round(1+np.random.rand(bs) * 9, 1)    # between 1-10, and one decimal place
                # GROSS -- hard coding some parameters here :/ 
                final_x = model.env.final.center[0]
                obs_x = (centers + final_x) / 2
                model.env.covers[0].center = np.expand_dims(np.stack([center_batch, 3.5 * np.ones_like(center_batch)], axis=1), 1)
                model.env.obs[0].center = np.expand_dims(np.stack([obs_x, 9. * np.ones_like(obs_x)], axis=1), 1)
                ic_imgs = torch.cat([convert_env_img_to_tensor("../figs/environments/%.1f"%cb) for cb in centers], dim=0)

                x_future, u_future = model.propagate_n(T, ic, ic_imgs)
                complete_traj = model.join_partial_future_signal(ic, x_future)      # [bs, time_dim, x_dim]

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

                writer.add_scalar('train/loss/state', loss_state, gradient_step)
                writer.add_scalar('train/loss/ctrl', loss_ctrl, gradient_step)
                writer.add_scalar('train/loss/STL', loss_stl, gradient_step)
                writer.add_scalar('train/loss/STL_true', loss_stl_true, gradient_step)
                writer.add_scalar('train/loss/total', loss, gradient_step)
                writer.add_scalar('train/parameters/teacher_training', teacher_training_value, gradient_step)
                writer.add_scalar('train/parameters/stl_scale', stl_scale_value, gradient_step)
                writer.add_scalar('train/parameters/weight_stl', weight_stl, gradient_step)

                # plotting progress
                if batch_idx % 20 == 0:
                    # trajectories from propagating initial states
                    traj_np = model.unstandardize_x(complete_traj).cpu().detach().numpy()
                    # trajectory from propagating initial state of expert trajectory
                    x_future, u_future = model.propagate_n(T, x_train[:,:1,:], imgs_train)
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
                    ax.set_ylim([-5, 15])

                    writer.add_figure('train/trajectory', fig1, gradient_step)

                    # controls plot
                    fig2, axs = plt.subplots(1,2,figsize=(15,6))
                    for (k,a) in enumerate(axs):
                        a.plot(u_train_.squeeze().cpu().detach().numpy()[:,k], label="Expert")
                        a.plot(model.unstandardize_u(u_future).squeeze().cpu().detach().numpy()[:,k], linestyle='--', label="Reconstruction")
                        a.grid()
                        a.set_xlim([0,T])
                        a.set_ylim([-4,4])
                    writer.add_figure('train/controls', fig2, gradient_step)


                if nan_flag:
                    # don't take step, wait for a new batch
                    continue
                loss.backward()
                optimizer.step()
                gradient_step += 1

            fig1.savefig(fig_dir + '/train/number={:02d}_iteration={:03d}.png'.format(number, train_iteration))

            torch.save({
                        'train_iteration': train_iteration,
                        'gradient_step': gradient_step,
                        'eval_iteration': eval_iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'ic': ic},
                        save_model_path + "/model_{:02d}_iteration={:03d}.png".format(number, train_iteration))

            # evaluation set
            model.eval()
            for (batch_idx, ic_) in enumerate(eval_loader):
              
                ic = ic_.to(device).float()        # [bs, 1, x_dim]
                model.eval()


                # reconstruct the expert model
                loss_state, loss_ctrl, x_traj_pred, u_traj_pred = model.reconstruction_loss(x_eval, u_eval, imgs_eval, tls_eval, teacher_training=1.0)
                loss_recon = loss_state + hps.weight_ctrl * loss_ctrl

                # with new ICs, propagate the trajectories
                bs = ic.shape[0]
                centers = np.round(1+np.random.rand(bs) * 9, 1)    # between 1-10, and one decimal place
                # GROSS -- hard coding some parameters here :/ 
                final_x = model.env.final.center[0]
                obs_x = (centers + final_x) / 2
                model.env.covers[0].center = np.expand_dims(np.stack([center_batch, 3.5 * np.ones_like(center_batch)], axis=1), 1)
                model.env.obs[0].center = np.expand_dims(np.stack([obs_x, 9. * np.ones_like(obs_x)], axis=1), 1)
                ic_imgs = torch.cat([convert_env_img_to_tensor("../figs/environments/%.1f"%cb) for cb in centers], dim=0)

                x_future, u_future = model.propagate_n(T, ic, ic_imgs)
                complete_traj = model.join_partial_future_signal(ic, x_future)      # [bs, time_dim, x_dim]

                # stl loss
                loss_stl = model.STL_loss(complete_traj, formula, formula_input_func, scale=-1)

                # total loss
                loss = hps.weight_recon * loss_recon + weight_stl * loss_stl

                writer.add_scalar('eval/state', loss_state, eval_iteration)
                writer.add_scalar('eval/ctrl', loss_ctrl, eval_iteration)
                writer.add_scalar('eval/STL', loss_stl, eval_iteration)
                writer.add_scalar('eval/total', loss, eval_iteration)


                traj_np = model.unstandardize_x(complete_traj).cpu().detach().numpy()
                # trajectory from propagating initial state of expert trajectory
                x_future, u_future = model.propagate_n(T, x_eval[:,:1,:])
                x_traj_prop = model.join_partial_future_signal(x_eval[:,:1,:], x_future)
                x_traj_prop = model.unstandardize_x(x_traj_prop).squeeze().detach().cpu().numpy()
                # trajectory from teacher training, and used for reconstruction loss (what the training sees)
                x_traj_pred = model.unstandardize_x(x_traj_pred)

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
                ax.set_ylim([-5, 15])

                writer.add_figure('eval/trajectory', fig1, eval_iteration)
                fig1.savefig(fig_dir + '/eval/number={:02d}_iteration={:03d}.png'.format(number, eval_iteration))

                fig2, axs = plt.subplots(1,2,figsize=(15,6))
                for (k,a) in enumerate(axs):
                    a.plot(u_train_.squeeze().cpu().detach().numpy()[:,k], label="Expert")
                    a.plot(model.unstandardize_u(u_future).squeeze().cpu().detach().numpy()[:,k], linestyle='--', label="Reconstruction")
                    a.grid()
                    a.set_xlim([0,T])
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

