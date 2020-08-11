import sys
import torch
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
import IPython
from torch.utils.data import Dataset, DataLoader
from learning import *

draw_params = {"initial": {"color": "lightskyblue", "fill": True, "alpha": 0.5}, "final": {"color": "coral", "fill": True, "alpha": 0.5}, "covers": {"color": "black", "fill": False}, "obs": {"color": "red", "fill": True, "alpha": 0.5} }


def train(model, train_traj, eval_traj, formula, formula_input_func, train_loader, eval_loader, device, tqdm, writer, hps, save_model_path, iter_max=np.inf):
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=hps.weight_decay, lr=hps.learning_rate)

    train_iteration = -1 # train iteration number
    eval_iteration = 0 # evaluation iteration number
    gradient_step = 0 # descent number
    kwargs = {}
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
    with tqdm(total=iter_max) as pbar:
        while True:
            train_iteration += 1 # i is num of runs through the data set
            if train_iteration == iter_max:
                return
            for (batch_idx, ic_) in enumerate(train_loader):
                
                optimizer.zero_grad()
                ic = torch.cat([x_train[:1,:,:], ic_.permute([1,0,2]).to(device).float()], dim=1)
                model.train()
                o, u, x_pred = model(x_train)

                # reconstruct the expert model
                loss_state, loss_ctrl = model.state_control_loss(x_train, x_train, u_train, teacher_training=hps.teacher_training(train_iteration))
                loss_recon = loss_state + hps.weight_ctrl * loss_ctrl
                
                # with new ICs, propagate the trajectories and keep them inside the reachable set
                x_future, u_future = model.propagate_n(T, ic)
                complete_traj = model.join_partial_future_signal(ic, x_future)
                loss_HJI = model.HJI_loss(complete_traj)
                
                # stl loss
                loss_stl = model.STL_loss(complete_traj, formula, lambda s,c: formula_input_func(s, c, device), scale=-1)
                
                # total loss
                loss = hps.weight_recon * loss_recon + hps.weight_hji(train_iteration) * loss_HJI + hps.weight_stl * loss_stl

                if torch.isnan(loss):
                    print("Encountered NaN!")
                    return
                else: 
                    save_model(model, optimizer, gradient_step, loss, save_model_path)


                writer.add_scalar('train/state', loss_state, gradient_step)
                writer.add_scalar('train/ctrl', loss_ctrl, gradient_step)
                writer.add_scalar('train/HJI', loss_HJI, gradient_step)
                writer.add_scalar('train/STL', loss_stl, gradient_step)
                writer.add_scalar('train/total', loss, gradient_step)
                writer.add_scalar('train/teacher_training', hps.teacher_training(train_iteration), gradient_step)


                if batch_idx % 20 == 0:
                    traj_np = unstandardize_data(complete_traj, mu_x, sigma_x).cpu().detach().numpy()
                    x_future, u_future = model.propagate_n(T, x_train[:1,:,:])
                    p = model.join_partial_future_signal(x_train[:1,:,:], x_future)
                    p = unstandardize_data(p, mu_x, sigma_x).squeeze().detach().cpu().numpy()
                    x_pred = unstandardize_data(x_pred, mu_x, sigma_x)
                    fig = plt.figure(figsize=(10,10))
                    model.env.draw2D(kwargs=draw_params)
                    plt.axis("equal")
                    plot_hji_contour()
                    plt.plot(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], linewidth=4)
                    plt.scatter(x_train_.squeeze().cpu().numpy()[:,0], x_train_.squeeze().cpu().numpy()[:,1], s=100)
                    plt.plot(p[:,0], p[:,1], linewidth=4)
                    plt.scatter(p[:,0], p[:,1], marker='^', s=100)
                    plt.plot(x_pred.cpu().detach().squeeze().numpy()[:,0], x_pred.cpu().detach().squeeze().numpy()[:,1])
                    plt.scatter(x_pred.cpu().detach().squeeze().numpy()[:,0], x_pred.cpu().detach().squeeze().numpy()[:,1], marker='*', s=100)
                    for j in range(traj_np.shape[1]):
                        plt.plot(traj_np[:,j,0], traj_np[:,j,1])
                        plt.scatter(traj_np[:,j,0], traj_np[:,j,1])

                    plt.xlim([-3, 12])
                    plt.ylim([-3, 12])
                    writer.add_figure('train/trajectory', fig, gradient_step)


                    fig = plt.figure(figsize=(15,6))
                    for k in [0,1]:
                        plt.subplot(1,2,k+1)
                        plt.plot(u_train_.squeeze().cpu().detach().numpy()[:,k])
                        plt.plot(unstandardize_data(u_future, mu_u, sigma_u).squeeze().cpu().detach().numpy()[:,k],'--')
                        plt.grid()
                        plt.xlim([0, 26])
                        plt.ylim([-4,4])
                    writer.add_figure('train/controls', fig, gradient_step)


                loss.backward()
                optimizer.step()
                gradient_step += 1

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
                loss_stl = model.STL_loss(complete_traj, formula, lambda s,c: formula_input_func(s, c, device), scale=-1)
                loss = hps.weight_recon * loss_recon + hps.weight_hji(train_iteration) * loss_HJI + hps.weight_stl * loss_stl
                
                traj_np = unstandardize_data(complete_traj, mu_x, sigma_x).cpu().detach().numpy()
                x_future, u_future = model.propagate_n(T, x_eval[:1,:,:])
                p = model.join_partial_future_signal(x_eval[:1,:,:], x_future)
                p = unstandardize_data(p, mu_x, sigma_x).squeeze().detach().cpu().numpy()
                
                writer.add_scalar('eval/state', loss_state, eval_iteration)
                writer.add_scalar('eval/ctrl', loss_ctrl, eval_iteration)
                writer.add_scalar('eval/HJI', loss_HJI, eval_iteration)
                writer.add_scalar('eval/STL', loss_stl, eval_iteration)
                writer.add_scalar('eval/total', loss, eval_iteration)
                x_pred = unstandardize_data(x_pred, mu_x, sigma_x)

                fig = plt.figure(figsize=(10,10))
                model.env.draw2D(kwargs=draw_params)
                plt.axis("equal")
                plot_hji_contour()
                plt.plot(x_eval_.squeeze().cpu().numpy()[:,0], x_eval_.squeeze().cpu().numpy()[:,1], linewidth=4)
                plt.scatter(x_eval_.squeeze().cpu().numpy()[:,0], x_eval_.squeeze().cpu().numpy()[:,1], s=100)
                plt.plot(p[:,0], p[:,1], linewidth=4)
                plt.scatter(p[:,0], p[:,1], marker='^', s=100)
                plt.plot(x_pred.squeeze().cpu().detach().numpy()[:,0], x_pred.squeeze().cpu().detach().numpy()[:,1])
                plt.scatter(x_pred.squeeze().cpu().detach().numpy()[:,0], x_pred.squeeze().cpu().detach().numpy()[:,1], marker='*', s=100)
                for j in range(traj_np.shape[1]):
                    plt.plot(traj_np[:,j,0], traj_np[:,j,1])
                    plt.scatter(traj_np[:,j,0], traj_np[:,j,1])

                plt.xlim([-3, 12])
                plt.ylim([-3, 12])
                writer.add_figure('eval/trajectory', fig, eval_iteration)
                
                
                fig = plt.figure(figsize=(15,6))
                for k in [0,1]:
                    plt.subplot(1,2,k+1)
                    plt.plot(u_eval_.squeeze().cpu().detach().numpy()[:,k])
                    plt.plot(unstandardize_data(u_future, mu_u, sigma_u).squeeze().cpu().detach().numpy()[:,k],'--')
                    plt.grid()
                    plt.xlim([0, 26])
                    plt.ylim([-4,4])
                writer.add_figure('eval/controls', fig, eval_iteration)

                eval_iteration += 1




            # Feel free to modify the progress bar
            pbar.set_postfix(loss='{:.2e}'.format(loss))
            pbar.update(1)
            # Log summaries
            # training progress

