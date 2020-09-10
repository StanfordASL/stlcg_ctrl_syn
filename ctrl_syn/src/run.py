import sys
sys.path.append('../../../stlcg_karen/src')
sys.path.append('../../expert_demo_ros/src/utils')
sys.path.append('..')

import stlcg
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
from train import train
from adversarial import adversarial

import IPython


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iter_max',  type=int, default=500, help="Number of training iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--lstm_dim',  type=int, default=32,    help="size of lstm_dim")
parser.add_argument('--device',    type=str, default="cpu",    help="cuda or cpu")
parser.add_argument('--dropout',   type=float, default="0.0",   help="dropout probability")
parser.add_argument('--weight_ctrl',   type=float, default="0.7",   help="weight on ctrl for recon")
parser.add_argument('--weight_recon',   type=float, default="1.0",   help="weight on recon")
parser.add_argument('--weight_hji',   type=float, default="-1",   help="weight on hji")
parser.add_argument('--weight_stl',   type=float, default="0.1",   help="weight on stl")
parser.add_argument('--teacher_training',   type=float, default="0.1",   help="probability of using previous output in rnn rollout")
parser.add_argument('--mode',   type=str, default="train",   help="train or eval")
parser.add_argument('--type',   type=str, default="goal",   help="goal or coverage")
parser.add_argument('--stl_scale',   type=float, default="1.0",   help="stlcg scaling parameter")
parser.add_argument('--status',   type=str, default="new",   help="new or continue(d) run")
parser.add_argument('--hji_max',   type=float, default="0.2",   help="maximum hji weight")
parser.add_argument('--hji_min',   type=float, default="0.0",   help="minimum hji weight")
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


layout = [
    ('run={:01d}', args.run),
    ('lstm_dim={:02d}', args.lstm_dim),
    ('dropout={:.2f}', args.dropout),
    ('teacher_training={:.1f}', args.teacher_training),
    ('weight_ctrl={:.1f}', args.weight_ctrl),
    ('weight_recon={:.1f}', args.weight_recon),
    ('weight_hji={:.1f}', args.weight_hji),
    ('hji_max={:.1f}', args.hji_max),
    ('hji_min={:.2f}', args.hji_min),
    ('weight_stl={:.1f}', args.weight_stl),
    ('stl_max={:.1f}', args.stl_max),
    ('stl_min={:.2f}', args.stl_min),
    ('stl_scale={:.1f}', args.stl_scale),
    ('scale_max={:.1f}', args.scale_max),
    ('scale_min={:.2f}', args.scale_min),
    ('iter_max={:.1f}', args.iter_max),
    ('device={:s}', args.device),
]


model_name = args.type + "_" + '_'.join([t.format(v) for (t, v) in layout])

runs_dir = "../runs/" + model_name
model_dir = "../models/" + model_name
log_dir = "../logs/" + model_name
fig_dir = "../figs/" + model_name

if (args.mode == "train") & os.path.exists(runs_dir):
    print("model exists already, changing run number")
    args.run += 1
    layout = [
        ('run={:01d}', args.run),
        ('lstm_dim={:02d}', args.lstm_dim),
        ('dropout={:.2f}', args.dropout),
        ('teacher_training={:.1f}', args.teacher_training),
        ('weight_ctrl={:.1f}', args.weight_ctrl),
        ('weight_recon={:.1f}', args.weight_recon),
        ('weight_hji={:.1f}', args.weight_hji),
        ('hji_max={:.1f}', args.hji_max),
        ('hji_min={:.2f}', args.hji_min),
        ('weight_stl={:.1f}', args.weight_stl),
        ('stl_max={:.1f}', args.stl_max),
        ('stl_min={:.2f}', args.stl_min),
        ('stl_scale={:.1f}', args.stl_scale),
        ('scale_max={:.1f}', args.scale_max),
        ('scale_min={:.2f}', args.scale_min),
        ('iter_max={:.1f}', args.iter_max),
        ('device={:s}', args.device),
    ]


    model_name = args.type + "_" + '_'.join([t.format(v) for (t, v) in layout])

    runs_dir = "../runs/" + model_name
    model_dir = "../models/" + model_name
    log_dir = "../logs/" + model_name
    fig_dir = "../figs/" + model_name

if args.action == "remove":
    try:
        os.remove(model_dir)
    except FileNotFoundError:
        print("model file does not exist")

    try:
        shutil.rmtree(runs_dir)
    except FileNotFoundError:
        print("runs folder does not exist")
        
    try:
        shutil.rmtree(fig_dir)
    except FileNotFoundError:
        print("figs folder does not exist")

    sys.exit() 
else:
    try:
        os.mkdir(fig_dir)
    except FileExistsError:
        print("fig folder already exists")

    try:
        os.mkdir(fig_dir + "/train")
    except FileExistsError:
        print("fig/train folder already exists")

    try:
        os.mkdir(fig_dir + "/eval")
    except FileExistsError:
        print("fig/eval folder already exists")

    try:
        os.mkdir(fig_dir + "/adversarial")
    except FileExistsError:
        print("fig/adversarial folder already exists")



write_log(log_dir, "\n\n" + model_name)

print(vars(args))
print('Model name:', model_name)

hps_names = ['weight_decay',
             'learning_rate',
             'teacher_training',
             'weight_ctrl',
             'weight_recon',
             'weight_hji',
             'weight_stl',
             'stl_scale',
             'adv_stl_scale',
             'alpha',
             'stl_type'
             ]


hyperparameters = namedtuple('hyperparameters', hps_names)

device = args.device
vf_cpu =  HJIValueFunction.apply
if device == "cuda":
    vf =  HJIValueFunction_cuda.apply
else:
    vf =  HJIValueFunction.apply



b = 80 / (1000.0/args.iter_max)
c = 6
if args.teacher_training >= 0.0:
    teacher_training = lambda ep: args.teacher_training
else:
    teacher_training = lambda ep: 0.1 + 0.9*np.exp(ep/b - c) / (1 + np.exp(ep/b - c))
    write_log(log_dir, "Teacher training: min={} max={} b={} c={}".format(0.1, 0.9, b, c))


if args.weight_hji >= 0.0:
    weight_hji = lambda ep: args.weight_hji
else:
    c1 = 8
    weight_hji = lambda ep: args.hji_min + (args.hji_max - args.hji_min) * np.exp(ep/b - c1) / (1 + np.exp(ep/b - c1))
    write_log(log_dir, "HJI weight: min={} max={} b={} c={}".format(args.hji_min, args.hji_max, b, c1))

if args.weight_stl >= 0.0:
    weight_stl = lambda ep: args.weight_stl
else:  
    weight_stl = lambda ep: args.stl_min + (args.stl_max - args.stl_min) * np.exp(ep/b - c) / (1 + np.exp(ep/b - c))
    write_log(log_dir, "STL weight: min={} max={} b={} c={}".format(args.stl_min, args.stl_max, b, c))

if args.stl_scale >= 0.0:
    stl_scale = lambda ep: args.stl_scale
else:
    stl_scale = lambda ep: args.scale_min + (args.scale_max - args.scale_min) * np.exp(ep/b - c) / (1 + np.exp(ep/b - c))
    write_log(log_dir, "STL scale: min={} max={} b={} c={}".format(args.scale_min, args.scale_max, b, c))
    
b0 = 80 / (1000.0/args.adv_iter_max)
c0 = 5

adv_stl_scale = lambda ep: args.scale_min + (args.scale_max - args.scale_min) * np.exp(ep/b0 - c0) / (1 + np.exp(ep/b0 - c0))
write_log(log_dir, "Adversarial stl scale: min={} max={} b={} c={}".format(args.scale_min, args.scale_max, b0, c0))


hps = hyperparameters(weight_decay=0.05, 
                      learning_rate=0.1, 
                      teacher_training=teacher_training,
                      weight_ctrl=args.weight_ctrl,
                      weight_recon=args.weight_recon,
                      weight_hji=weight_hji,
                      weight_stl=weight_stl,
                      stl_scale=stl_scale,
                      adv_stl_scale=adv_stl_scale,
                      alpha=0.001,
                      stl_type=args.type)




# original data
x_train_, u_train_, stats = prepare_data("../../hji/data/" + args.type + "/expert_traj_train.npy")
x_eval_, u_eval_, _ = prepare_data("../../hji/data/" + args.type + "/expert_traj_eval.npy")

ic_train_ = torch.Tensor(InitialConditionDataset(args.trainset_size, vf_cpu, args.type)).float()
ic_eval_ = torch.tensor(InitialConditionDataset(args.evalset_size, vf_cpu, args.type)).float()


# standardized data
x_train = standardize_data(x_train_, stats[0][:,:,:4], stats[1][:,:,:4])
u_train = standardize_data(u_train_, stats[0][:,:,4:], stats[1][:,:,4:])

x_eval = standardize_data(x_eval_, stats[0][:,:,:4], stats[1][:,:,:4])
u_eval = standardize_data(u_eval_, stats[0][:,:,4:], stats[1][:,:,4:])

ic_train = standardize_data(ic_train_, stats[0][:,:,:4], stats[1][:,:,:4])
ic_eval = standardize_data(ic_eval_, stats[0][:,:,:4], stats[1][:,:,:4])


ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=args.trainset_size//32, shuffle=True)
ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=args.evalset_size, shuffle=False)


state_dim = ic_eval.shape[-1]
ctrl_dim = u_eval.shape[-1]

if args.type == "coverage":
    params = {  "covers": [Box([0., 6.],[4, 8.]), Box([6., 2.],[10.0, 4.0])],
                "obstacles": [Circle([7., 7.], 2.0)],
                "initial": Box([-1., -1.],[1., 1.]),
                "final": Box([9.0, 9.0],[11.0, 11.0])
           }
elif args.type == "goal":
    params = {  "covers": [],
                "obstacles": [Circle([7., 7.], 2.0)],
                "initial": Box([-1., -1.],[1., 1.]),
                "final": Box([9.0, 9.0],[11.0, 11.0])
                }

elif args.type == "test":
    params = {  "covers": [],
                "obstacles": [],
                "initial": Box([-1., -1.],[1., 1.]),
                "final": Box([4.0, 4.0],[6.0, 6.0])
           } 
elif args.type == "coverage_test":
    params = {  "covers": [Box([0., 6.],[4, 8.])],
                "obstacles": [Circle([7., 7.], 2.0)],
                "initial": Box([-1., -1.],[1., 1.]),
                "final": Box([9.0, 9.0],[11.0, 11.0])
           } 
env = Environment(params)



stl_traj = x_train_.permute([1,0,2]).flip(1)
if len(params["obstacles"]) > 0:
    obs_avoid, obs_avoid_input = outside_circle_stl(stl_traj, env.obs[0], device)

in_end_box, in_end_box_input = in_box_stl(stl_traj, env.final, device)
stop_in_end_box, stop_in_end_box_input = stop_in_box_stl(stl_traj, env.final, device)

end_goal = stlcg.Eventually(subformula=stlcg.Always(subformula=stop_in_end_box))

if len(params["covers"]) == 2:
    in_cov_1, in_cov_1_input = in_box_stl(stl_traj, env.covers[0], device)
    in_cov_2, in_cov_2_input = in_box_stl(stl_traj, env.covers[1], device)
    coverage_1 = stlcg.Eventually(subformula=stlcg.Always(subformula=in_cov_1, interval=[0, 10]))
    coverage_2 = stlcg.Eventually(subformula=stlcg.Always(subformula=in_cov_2, interval=[0, 10]))

if len(params["covers"]) == 1:
    in_cov_1, in_cov_1_input = in_box_stl(stl_traj, env.covers[0], device)
    coverage_1 = stlcg.Eventually(subformula=stlcg.Always(subformula=in_cov_1, interval=[0, 10]))

if args.type == "goal":
    formula = obs_avoid & end_goal
    get_formula_input = get_goal_formula_input
elif args.type == "coverage":
    coverage_stl = stlcg.Until(subformula1=coverage_1, subformula2=coverage_2)
    formula = stlcg.Until(subformula1=coverage_stl, subformula2=end_goal) & obs_avoid
    get_formula_input = get_coverage_formula_input
elif args.type == "test":
    formula = end_goal
    get_formula_input = get_test_formula_input
elif args.type == "coverage_test":
    formula = stlcg.Until(subformula1=coverage_1, subformula2=end_goal) & obs_avoid
    get_formula_input = get_coverage_test_formula_input
else:
    raise NameError(args.type + " is not defined.")


model = STLPolicy(kinematic_bicycle, state_dim, ctrl_dim, args.lstm_dim, stats, env, vf, args.dropout).to(device)


if args.mode == "train":
    train(model=model,
         train_traj=(x_train, u_train),
         eval_traj=(x_eval, u_eval),
         formula=formula,
         formula_input_func=get_formula_input,
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
    adv_ic = adversarial(model=model, 
                         T=x_train.shape[0]+4, 
                         formula=formula,
                         formula_input_func=get_formula_input,
                         device=device,
                         tqdm=tqdm.tqdm, 
                         writer=SummaryWriter(log_dir=runs_dir), 
                         hps=hps, 
                         save_model_path=model_dir, 
                         number=args.number,
                         iter_max=args.adv_iter_max,
                         adv_n_samples=64)

    breakpoint()
    prompt = input("Adversarial training finished, final comments:")
    write_log(log_dir, "Adversarial: done! {}".format(prompt))
elif args.mode == "adv_training_iteration":

    for _ in range(3):
        train(model=model,
             train_traj=(x_train, u_train),
             eval_traj=(x_eval, u_eval),
             formula=formula,
             formula_input_func=get_formula_input,
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
                             formula_input_func=get_formula_input,
                             device=device,
                             tqdm=tqdm.tqdm, 
                             writer=SummaryWriter(log_dir=runs_dir), 
                             hps=hps, 
                             save_model_path=model_dir, 
                             number=_,
                             iter_max=args.adv_iter_max,
                             adv_n_samples=32)
        write_log(log_dir, "Adversarial: {} adversarial phase(s) done".format(_+1))
        write_log(log_dir, "Adversarial: {} adversarial examples".format(adv_ic.shape[0]))

        train_num = adv_ic.shape[0]//2
        if train_num == 0:
            ValueError("Not enough adversarial examples")

        ic_train_ = torch.Tensor(InitialConditionDataset(args.trainset_size, vf_cpu, args.type)).float()     # [bs, 1, x_dim]
        ic_eval_ = torch.tensor(InitialConditionDataset(args.evalset_size, vf_cpu, args.type)).float()       # [bs, 1, x_dim]

        ic_adv_train_ = torch.cat([ic_train_, adv_ic[:train_num,:,:]], dim=0).to(device)
        ic_adv_eval_ = torch.cat([ic_eval_, adv_ic[train_num:,:,:]], dim=0).to(device)

        ic_train = standardize_data(ic_adv_train_, stats[0][:,:,:4], stats[1][:,:,:4])
        ic_eval = standardize_data(ic_adv_eval_, stats[0][:,:,:4], stats[1][:,:,:4])


        ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=args.trainset_size//32, shuffle=True)
        ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=args.evalset_size, shuffle=False)

    prompt = input("Adversarial and training loop finished, final comments:")
    write_log(log_dir, "Adversarial + training: done! {}".format(prompt))

else:
    raise NameError("args.mode undefined")
