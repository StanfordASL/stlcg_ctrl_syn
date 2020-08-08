import sys
sys.path.append('../../../stlcg_karen/src')
sys.path.append('../../expert_demo_ros/src/utils')
sys.path.append('..')

import stlcg
import matplotlib.pyplot as plt
import numpy as np
import os
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
from train import train



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iter_max',  type=int, default=500, help="Number of training iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--lstm_dim',  type=int, default=32,    help="size of lstm_dim")
parser.add_argument('--device',    type=str, default="cpu",    help="cuda or cpu")
parser.add_argument('--dropout',   type=float, default="0.0",   help="dropout probability")
parser.add_argument('--weight_ctrl',   type=float, default="0.7",   help="weight on ctrl for recon")
parser.add_argument('--weight_recon',   type=float, default="1.0",   help="weight on recon")
parser.add_argument('--weight_hji',   type=float, default="0.1",   help="weight on hji")
parser.add_argument('--weight_stl',   type=float, default="0.1",   help="weight on stl")
parser.add_argument('--teacher_training',   type=float, default="0.1",   help="probability of using previous output in rnn rollout")

args = parser.parse_args()


layout = [
    ('run={:01d}', args.run),
    ('lstm_dim={:02d}', args.lstm_dim),
    ('dropout={:.2f}', args.dropout),
    ('teacher_training={:.1f}', args.teacher_training),
    ('weight_ctrl={:.1f}', args.weight_ctrl),
    ('weight_recon={:.1f}', args.weight_recon),
    ('weight_hji={:.1f}', args.weight_hji),
    ('weight_stl={:.1f}', args.weight_stl),
    ('device={:s}', args.device),
]


model_name = "policy_" + '_'.join([t.format(v) for (t, v) in layout])
print(vars(args))
print('Model name:', model_name)

hps_names = ['weight_decay',
             'learning_rate',
             'teacher_training',
             'weight_ctrl',
             'weight_recon',
             'weight_hji',
             'weight_stl'
             ]


hyperparameters = namedtuple('hyperparameters', hps_names)

device = args.device
vf_cpu =  HJIValueFunction.apply
if device == "cuda":
    vf =  HJIValueFunction_cuda.apply
else:
    vf =  HJIValueFunction.apply

b = 50 / (1000.0/args.iter_max)
c = 10
sigmoid = lambda ep: 0.1 + 0.9*np.exp(ep/b - c) / (1 + np.exp(ep/b - c))
if args.teacher_training >= 0.0:
    teacher_training = lambda ep: args.teacher_training
else:
    teacher_training = sigmoid

hps = hyperparameters(weight_decay=0.1, 
                      learning_rate=0.001, 
                      teacher_training=teacher_training,
                      weight_ctrl=args.weight_ctrl,
                      weight_recon=args.weight_recon,
                      weight_hji=args.weight_hji,
                      weight_stl=args.weight_stl)




# original data
x_train_, u_train_, stats = prepare_data("../../hji/data/expert_traj_train.npy")
x_eval_, u_eval_, _ = prepare_data("../../hji/data/expert_traj_eval.npy")

ic_train_ = torch.Tensor(InitialConditionDataset(2048, vf_cpu)).float()
ic_eval_ = torch.tensor(InitialConditionDataset(128, vf_cpu)).float()


# standardized data
x_train = standardize_data(x_train_, stats[0][:,:,:4], stats[1][:,:,:4])
u_train = standardize_data(u_train_, stats[0][:,:,4:], stats[1][:,:,4:])

x_eval = standardize_data(x_eval_, stats[0][:,:,:4], stats[1][:,:,:4])
u_eval = standardize_data(u_eval_, stats[0][:,:,4:], stats[1][:,:,4:])

ic_train = standardize_data(ic_train_, stats[0][:,:,:4], stats[1][:,:,:4])
ic_eval = standardize_data(ic_eval_, stats[0][:,:,:4], stats[1][:,:,:4])


ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=32, shuffle=True)
ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=ic_eval.shape[0], shuffle=False)


state_dim = ic_eval.shape[-1]
ctrl_dim = u_eval.shape[-1]

params = {  "covers": [Box([0., 6.],[4, 8.]), Box([6., 2.],[10.0, 4.0])],
            "obstacles": [Circle([7., 7.], 2.0)],
            "initial": Box([-1., -1.],[1., 1.]),
            "final": Box([9.0, 9.0],[11.0, 11.0])
       }

cov_env = CoverageEnv(params)



stl_traj = x_train.permute([1,0,2]).flip(1)

obs_avoid, obs_avoid_input = outside_circle_stl(stl_traj, cov_env.obs[0], device)
in_box, in_box_input = in_box_stl(stl_traj, cov_env.final, device)
end_goal = stlcg.Eventually(subformula=stlcg.Always(subformula=in_box))

formula = obs_avoid & end_goal



model = STLPolicy(kinematic_bicycle, state_dim, ctrl_dim, args.lstm_dim, stats, cov_env, vf, args.dropout).to(device)

log_dir = "../runs/" + model_name
model_dir = "../models/" + model_name

train(model=model,
     train_traj=(x_train, u_train),
     eval_traj=(x_eval, u_eval),
     formula=formula,
     formula_input_func=get_formula_input,
     train_loader=ic_trainloader,
     eval_loader=ic_evalloader,
     device=device,
     tqdm=tqdm.tqdm,
     writer=SummaryWriter(log_dir=log_dir),
     hps=hps,
     save_model_path=model_dir,
     iter_max=args.iter_max
     )

