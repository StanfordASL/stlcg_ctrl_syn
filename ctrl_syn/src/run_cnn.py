import sys
sys.path.append('../../../stlcg_karen/src')
sys.path.append('../../expert_demo_ros/src/utils')
sys.path.append('..')

import stlcg
import stlviz
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
import scipy.io as sio

import tqdm
import argparse
from collections import namedtuple

from torch.utils.data import Dataset, DataLoader
from torch_interpolations.torch_interpolations.multilinear import RegularGridInterpolator
from torch.utils.tensorboard import SummaryWriter

from environment import *
from src.learning import *
from learning import *
from utils import *
from stl import *
from train import train
from test import test
from adversarial import adversarial

import IPython

dt = 0.5

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iter_max',  type=int, default=500, help="Number of training iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--lstm_dim',  type=int, default=32,    help="size of lstm_dim")
parser.add_argument('--device',    type=str, default="cpu",    help="cuda or cpu")
parser.add_argument('--dropout',   type=float, default="0.0",   help="dropout probability")
parser.add_argument('--weight_ctrl',   type=float, default="0.7",   help="weight on ctrl for recon")
parser.add_argument('--weight_recon',   type=float, default="1.0",   help="weight on recon")
parser.add_argument('--weight_stl',   type=float, default="0.1",   help="weight on stl")
parser.add_argument('--teacher_training',   type=float, default="0.1",   help="probability of using previous output in rnn rollout")
parser.add_argument('--mode',   type=str, default="train",   help="train or eval, or test for debugging")
parser.add_argument('--type',   type=str, default="coverage",   help="goal or coverage")
parser.add_argument('--stl_scale',   type=float, default="1.0",   help="stlcg scaling parameter")
parser.add_argument('--status',   type=str, default="new",   help="new or continue(d) run")
parser.add_argument('--stl_max',   type=float, default="0.2",   help="maximum stl weight")
parser.add_argument('--stl_min',   type=float, default="0.1",   help="minimum stl weight")
parser.add_argument('--scale_max',   type=float, default="50.0",   help="maximum scale weight")
parser.add_argument('--scale_min',   type=float, default="0.1",   help="minimum scale weight")
parser.add_argument('--trainset_size',   type=int, default="1024",   help="size of training ic set")
parser.add_argument('--evalset_size',   type=int, default="64",   help="size of eval ic set")
parser.add_argument('--number',   type=int, default="0",   help="train iteration number, used to reset annealing schedules at each new iteration")
parser.add_argument('--action',   type=str, default="create",   help="run or remove")
parser.add_argument('--adv_iter_max',   type=int, default="100",   help="number of adversarial steps")



args = parser.parse_args()
device = args.device

layout = [
    ('run={:01d}', args.run),
    ('lstm_dim={:02d}', args.lstm_dim),
    ('teacher_training={:.1f}', args.teacher_training),
    ('weight_ctrl={:.1f}', args.weight_ctrl),
    ('weight_recon={:.1f}', args.weight_recon),
    ('weight_stl={:.1f}', args.weight_stl),
    ('stl_max={:.1f}', args.stl_max),
    ('stl_min={:.2f}', args.stl_min),
    ('stl_scale={:.1f}', args.stl_scale),
    ('scale_max={:.1f}', args.scale_max),
    ('scale_min={:.2f}', args.scale_min),
    ('iter_max={:.1f}', args.iter_max),
]


model_name = args.type + "_" + '_'.join([t.format(v) for (t, v) in layout])

runs_dir = "../runs/" + model_name
model_dir = "../models/" + model_name
log_dir = "../logs/" + model_name
fig_dir = "../figs/" + model_name
adv_dir = "../adv/" + model_name
nan_dir = "../nan/" + model_name

if (args.mode == "train") & os.path.exists(runs_dir) & (args.action != "remove"):
    print("model exists already, changing run number")
    args.run += 1
    layout = [
        ('run={:01d}', args.run),
        ('lstm_dim={:02d}', args.lstm_dim),
        ('teacher_training={:.1f}', args.teacher_training),
        ('weight_ctrl={:.1f}', args.weight_ctrl),
        ('weight_recon={:.1f}', args.weight_recon),
        ('weight_stl={:.1f}', args.weight_stl),
        ('stl_max={:.1f}', args.stl_max),
        ('stl_min={:.2f}', args.stl_min),
        ('stl_scale={:.1f}', args.stl_scale),
        ('scale_max={:.1f}', args.scale_max),
        ('scale_min={:.2f}', args.scale_min),
        ('iter_max={:.1f}', args.iter_max),
    ]

    model_name = args.type + "_" + '_'.join([t.format(v) for (t, v) in layout])

    runs_dir = "../runs/" + model_name
    model_dir = "../models/" + model_name
    log_dir = "../logs/" + model_name
    fig_dir = "../figs/" + model_name
    adv_dir = "../adv/" + model_name
    nan_dir = "../nan/" + model_name

if args.action == "remove":

    ans = input("Are you sure you want to remove the files? (y/n) ")
    if ans == "y":
        remove_directory(model_dir)
        remove_directory(runs_dir)
        remove_directory(fig_dir)
        remove_directory(adv_dir)
        remove_directory(nan_dir)
    else: sys.exit()
else:
    make_directory(model_dir)
    make_directory(fig_dir)
    make_directory(fig_dir + "/train")
    make_directory(fig_dir + "/eval")
    make_directory(fig_dir + "/adversarial")
    make_directory(adv_dir)
    make_directory(nan_dir)

write_log(log_dir, "\n\n" + model_name)

prompt = input("Additional model information: ")
write_log(log_dir, "Additional model info: {}".format(prompt))


print(vars(args))
print('Model name:', model_name)

hps_names = ['weight_decay',
             'learning_rate',
             'teacher_training',
             'weight_ctrl',
             'weight_recon',
             'weight_stl',
             'stl_scale',
             'adv_stl_scale',
             'alpha',
             'stl_type',
             ]


hyperparameters = namedtuple('hyperparameters', hps_names)

b = 80 / (1000.0/args.iter_max)
c = 6
if args.teacher_training >= 0.0:
    teacher_training = lambda ep: args.teacher_training
else:
    teacher_training = lambda ep: sigmoidal_anneal(ep, 0.1, 0.9, b, c)
    write_log(log_dir, "Teacher training: min={} max={} b={} c={}".format(0.1, 0.9, b, c))


if args.weight_stl >= 0.0:
    weight_stl = lambda ep: args.weight_stl
else:
    weight_stl = lambda ep: sigmoidal_anneal(ep, args.stl_min, args.stl_max, b, c)
    write_log(log_dir, "STL weight: min={} max={} b={} c={}".format(args.stl_min, args.stl_max, b, c))

if args.stl_scale >= 0.0:
    stl_scale = lambda ep: args.stl_scale
else:
    stl_scale = lambda ep: sigmoidal_anneal(ep, args.scale_min, args.scale_max, b, c)
    write_log(log_dir, "STL scale: min={} max={} b={} c={}".format(args.scale_min, args.scale_max, b, c))
    
b0 = 80 / (1000.0/args.adv_iter_max)
c0 = 5

adv_stl_scale = lambda ep: sigmoidal_anneal(ep, args.scale_min, args.scale_max, b0, c0)
write_log(log_dir, "Adversarial stl scale: min={} max={} b={} c={}".format(args.scale_min, args.scale_max, b0, c0))


hps = hyperparameters(weight_decay=0.05, 
                      learning_rate=0.1, 
                      teacher_training=teacher_training,
                      weight_ctrl=args.weight_ctrl,
                      weight_recon=args.weight_recon,
                      weight_stl=weight_stl,
                      stl_scale=stl_scale,
                      adv_stl_scale=adv_stl_scale,
                      alpha=0.001,
                      stl_type=args.type)




# original data

x_train_, u_train_, tls_train, imgs_train, stats = prepare_data_img("../expert/coverage_cnn/train")
x_eval_, u_eval_, tls_eval, imgs_eval, stats = prepare_data_img("../expert/coverage_cnn/eval")

#  setting up environment
# will be overridden later
params = 
center_batch = np.round(1+np.random.rand(16) * 9, 1)
final_x = 5.0
obs_x = (center_batch + final_x) / 2

params = { "covers": [Circle(np.expand_dims(np.stack([center_batch, 3.5 * np.ones_like(center_batch)], axis=1), 1), 2.0)],
   "obstacles": [Circle(np.expand_dims(np.stack([obs_x, 9. * np.ones_like(obs_x)], axis=1), 1), 1.5)],
   "initial": Box([2, -4.],[8, -2]),
   "final": Circle([final_x, 13], 1.0)
} 
env = Environment(params)
             

# initial conditions set
lower = torch.tensor([env.initial.lower[0], env.initial.lower[1], np.pi/4, 0])
upper = torch.tensor([env.initial.upper[0], env.initial.upper[1], 3*np.pi/4, 2])

ic_train_ = initial_conditions(args.trainset_size, lower, upper)
ic_eval_ = initial_conditions(args.evalset_size, lower, upper)


# standardized data
x_train = standardize_data(x_train_, stats[0][:,:,:4], stats[1][:,:,:4])
u_train = standardize_data(u_train_, stats[0][:,:,4:], stats[1][:,:,4:])

x_eval = standardize_data(x_eval_, stats[0][:,:,:4], stats[1][:,:,:4])
u_eval = standardize_data(u_eval_, stats[0][:,:,4:], stats[1][:,:,4:])

ic_train = standardize_data(ic_train_, stats[0][:,:,:4], stats[1][:,:,:4])
ic_eval = standardize_data(ic_eval_, stats[0][:,:,:4], stats[1][:,:,:4])


ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=args.trainset_size//32, shuffle=True)
ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=args.evalset_size, shuffle=False)


in_end_goal = inside_circle(env.final, "distance to final")
stop_in_end_goal = in_end_goal & (stlcg.Expression('speed')  < 0.5)
end_goal = stlcg.Eventually(subformula=stlcg.Always(stop_in_end_goal))
coverage = stlcg.Eventually(subformula=(always_inside_circle(env.covers[0], "distance to coverage", interval=[0,10]) & (stlcg.Expression('speed')  < 1.0)))
avoid_obs = always_outside_circle(env.obs[0], "distance to obstacle")

formula = stlcg.Until(subformula1=coverage, subformula2=end_goal) & avoid_obs
stl_graph = stlviz.make_stl_graph(formula)
stlviz.save_graph(stl_graph, fig_dir + "/stl")

dynamics = KinematicBicycle(dt)
model = STLCNNPolicy(dynamics, 
                  args.lstm_dim, 
                  stats, 
                  env).to(device)


if args.mode == "test":
    test(model=model, 
         train_traj=(x_train, u_train), 
         formula=formula, 
         formula_input_func=lambda s: get_formula_input(s, env.covers[0], env.obs[0], env.final, device, backwards=False),
         train_loader=ic_trainloader,
         device=device)

    write_log(log_dir, "Testing: done!")

elif args.mode == "train":
    train(model=model,
         train_traj=(x_train, u_train),
         eval_traj=(x_eval, u_eval),
         formula=formula,
         formula_input_func=lambda s: get_formula_input(s, env.covers[0], env.obs[0], env.final, device, backwards=False),
         train_loader=ic_trainloader,
         eval_loader=ic_evalloader,
         device=device,
         tqdm=tqdm.tqdm,
         writer=SummaryWriter(log_dir=runs_dir),
         hps=hps,
         save_model_path=model_dir,
         number=args.number,
         iter_max=args.iter_max,
         status=args.status
         )

    prompt = input("Training finished, final comments:")
    write_log(log_dir, "Training: done! {}".format(prompt))

elif args.mode == 'eval':
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    IPython.embed(banner1="Model loaded in evaluation mode.")


elif args.mode == "adversarial":
    # adv_ic is [bs, 1, x_dim], cpu in unstandardized form

    adv_ic = adversarial(model=model, 
                         T=x_train.shape[1]+4, 
                         formula=formula,
                         formula_input_func=lambda s: get_formula_input(s, env.covers[0], env.obs[0], env.final, device, backwards=False),
                         device=device,
                         tqdm=tqdm.tqdm, 
                         writer=SummaryWriter(log_dir=runs_dir), 
                         hps=hps, 
                         save_model_path=model_dir, 
                         number=args.number,
                         lower=lower,
                         upper=upper,
                         iter_max=args.adv_iter_max,
                         adv_n_samples=64)

    np.save(adv_dir + "/number={}".format(args.number), adv_ic.detach().numpy())
    write_log(log_dir, "Adversarial: Saved adversarial initial conditions in npy file number={}".format(args.number))
    breakpoint()
    prompt = input("Adversarial training finished, final comments:")
    write_log(log_dir, "Adversarial: done! {}".format(prompt))



elif args.mode == "adv_training_iteration":

    if args.status == "continue":
        write_log(log_dir, "Adversarial: Continuing adversarial training at number={}".format(args.number))    
        start_idx = args.number
        adv_ic = torch.tensor(np.load(adv_dir + "/number={}.npy".format(args.number - 1)))
        write_log(log_dir, "Adversarial: Loaded adversarial initial conditions from previous adversarial training number={}".format(args.number - 1))    
        train_num = (adv_ic.shape[0] // 3) * 2
        if train_num == 0:
            write_log(log_dir, "Adversarial: Not enough adversarial examples.")
            ValueError("Not enough adversarial examples")
        eval_num = adv_ic.shape[0] - train_num

        ic_train_ = torch.Tensor(InitialConditionDataset(args.trainset_size - train_num, vf_cpu, args.type)).float()     # [bs, 1, x_dim]
        ic_eval_ = torch.tensor(InitialConditionDataset(args.evalset_size - eval_num, vf_cpu, args.type)).float()       # [bs, 1, x_dim]
        ic_adv_train_ = torch.cat([ic_train_, adv_ic[:train_num,:,:]], dim=0).to(stats[0].device)   # train_num 
        ic_adv_eval_ = torch.cat([ic_eval_, adv_ic[train_num:,:,:]], dim=0).to(stats[0].device)     # eval_num

        ic_train = standardize_data(ic_adv_train_, stats[0][:,:,:4], stats[1][:,:,:4]).to(device)
        ic_eval = standardize_data(ic_adv_eval_, stats[0][:,:,:4], stats[1][:,:,:4]).to(device)

        ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=args.trainset_size//32, shuffle=True)
        ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=args.evalset_size, shuffle=False)

    else:
        start_idx = 0

    for _ in range(start_idx, start_idx + 3):
        train(model=model,
             train_traj=(x_train, u_train),
             eval_traj=(x_eval, u_eval),
             formula=formula,
             formula_input_func=lambda s, c: get_formula_input(s,c, device),
             train_loader=ic_trainloader,
             eval_loader=ic_evalloader,
             device=device,
             tqdm=tqdm.tqdm,
             writer=SummaryWriter(log_dir=runs_dir),
             hps=hps,
             save_model_path=model_dir,
             number=_,
             iter_max=(args.iter_max)*(1 + _),
             status=args.status if (_== 0) else "continue"
             )
        write_log(log_dir, "Training: {} training phase(s) done".format(_+1))

        # adv_ic is [bs, 1, x_dim], cpu in unstandardized form
        adv_ic = adversarial(model=model, 
                             T=x_train.shape[0]+4, 
                             formula=formula,
                             formula_input_func=lambda s, c: get_formula_input(s,c, device),
                             device=device,
                             tqdm=tqdm.tqdm, 
                             writer=SummaryWriter(log_dir=runs_dir), 
                             hps=hps, 
                             save_model_path=model_dir, 
                             number=_,
                             iter_max=args.adv_iter_max,
                             adv_n_samples=32)
        np.save(adv_dir + "/number={}".format(_), adv_ic.detach().numpy())
        write_log(log_dir, "Adversarial: Saved adversarial initial conditions in npy file number={}".format(_))    
        write_log(log_dir, "Adversarial: {} adversarial phase(s) done".format(_+1))
        write_log(log_dir, "Adversarial: {} adversarial examples".format(adv_ic.shape[0]))

        train_num = (adv_ic.shape[0] // 3) * 2
        if train_num == 0:
            write_log(log_dir, "Adversarial: Not enough adversarial examples.")
            ValueError("Not enough adversarial examples")
        eval_num = adv_ic.shape[0] - train_num

        ic_train_ = torch.Tensor(InitialConditionDataset(args.trainset_size - train_num, vf_cpu, args.type)).float()     # [bs, 1, x_dim]
        ic_eval_ = torch.tensor(InitialConditionDataset(args.evalset_size - eval_num, vf_cpu, args.type)).float()       # [bs, 1, x_dim]

        ic_adv_train_ = torch.cat([ic_train_, adv_ic[:train_num,:,:]], dim=0).to(stats[0].device)   # train_num 
        ic_adv_eval_ = torch.cat([ic_eval_, adv_ic[train_num:,:,:]], dim=0).to(stats[0].device)     # eval_num

        ic_train = standardize_data(ic_adv_train_, stats[0][:,:,:4], stats[1][:,:,:4]).to(device)
        ic_eval = standardize_data(ic_adv_eval_, stats[0][:,:,:4], stats[1][:,:,:4]).to(device)


        ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=args.trainset_size//32, shuffle=True)
        ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=args.evalset_size, shuffle=False)

    prompt = input("Adversarial and training loop finished, final comments:")
    write_log(log_dir, "Adversarial + training: done! {}".format(prompt))

else:
    raise NameError("args.mode undefined")
