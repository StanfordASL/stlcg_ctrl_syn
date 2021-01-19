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
from train import train_cnn
from test import test
from adversarial import *

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
parser.add_argument('--trainset_size',   type=int, default="128",   help="size of training ic set")
parser.add_argument('--evalset_size',   type=int, default="64",   help="size of eval ic set")
parser.add_argument('--number',   type=int, default="0",   help="train iteration number, used to reset annealing schedules at each new iteration")
parser.add_argument('--action',   type=str, default="create",   help="run or remove")
parser.add_argument('--adv_iter_max',   type=int, default="100",   help="number of adversarial steps")
parser.add_argument('--adv_n_samples',   type=int, default="128",   help="number of adversarial steps")

parser.add_argument('--img_bs',   type=int, default=8,   help="number of different evironments per batch")
parser.add_argument('--expert_mini_bs',   type=int, default=4,   help="number of expert demonstrations per environment")


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
elif args.action != "test":
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
             'img_bs',
             'expert_mini_bs'
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
                      img_bs=args.img_bs,
                      expert_mini_bs=args.expert_mini_bs)



# original data

x_train_, u_train_, tls_train, imgs_train, stats, centers_train = prepare_data_img("../expert/coverage_cnn/train")
x_eval_, u_eval_, tls_eval, imgs_eval, stats, centers_eval = prepare_data_img("../expert/coverage_cnn/eval")

#  setting up environment
# will be overridden later
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




in_end_goal = inside_circle(env.final, "distance to final")
stop_in_end_goal = in_end_goal & (stlcg.Expression('speed')  < 0.5)
end_goal = stlcg.Eventually(subformula=stlcg.Always(subformula=stop_in_end_goal, interval=[0,4]))

in_coverage = inside_circle(env.covers[0], "distance to coverage") & (stlcg.Expression('speed')  < 2.0)
coverage = stlcg.Eventually(subformula=stlcg.Always(in_coverage, interval=[0,8]))

avoid_obs = always_outside_circle(env.obs[0], "distance to obstacle")

formula = stlcg.Until(subformula1=coverage, subformula2=end_goal) & avoid_obs

formula_input_func=lambda s, cov, obs: get_formula_input(s, cov, obs, env.final, device, backwards=False)
stl_graph = stlviz.make_stl_graph(formula)
stlviz.save_graph(stl_graph, fig_dir + "/stl")

params={"lr" : 0.7, "lf" : 0.5, "V_min" : 0.0, "V_max" : 5.0, "a_min" : -3, "a_max" : 3, "delta_min" : -0.344, "delta_max" : 0.344, "disturbance_scale" : [0.05, 0.02]}

dynamics = KinematicBicycle(dt, params)
model = STLCNNPolicy(dynamics,
                  args.lstm_dim,
                  stats,
                  env).to(device)
model.switch_device(device)


if args.mode == "test":

    n_train = 64
    n_eval = 16

    ic_train_ = initial_conditions(n_train, lower, upper)
    ic_eval_ = initial_conditions(n_eval, lower, upper)

    # standardized data
    x_train = model.standardize_x(x_train_.to(device).float())
    u_train = model.standardize_u(u_train_.to(device).float())

    x_eval = model.standardize_x(x_eval_.to(device).float())
    u_eval = model.standardize_u(u_eval_.to(device).float())

    ic_train = model.standardize_x(ic_train_.to(device).float())
    ic_eval = model.standardize_x(ic_eval_.to(device).float())



    ic_train = InitialConditionDatasetCNN(ic_train)
    ic_eval = InitialConditionDatasetCNN(ic_eval)

    ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=n_train//8, shuffle=True)
    ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=n_eval, shuffle=False)

    hps = hyperparameters(weight_decay=0.05,
                      learning_rate=0.1,
                      teacher_training=teacher_training,
                      weight_ctrl=args.weight_ctrl,
                      weight_recon=args.weight_recon,
                      weight_stl=weight_stl,
                      stl_scale=stl_scale,
                      adv_stl_scale=1.0,
                      alpha=0.001,
                      img_bs=4,
                      expert_mini_bs=args.expert_mini_bs)

    train_cnn(model=model,
              train_traj=(x_train, u_train),
              eval_traj=(x_eval, u_eval),
              imgs=(imgs_train, imgs_eval),
              tls=(tls_train, tls_eval),
              centers=(centers_train, centers_eval),
              formula=formula,
              formula_input_func=formula_input_func,
              train_loader=ic_trainloader,
              eval_loader=ic_evalloader,
              device=device,
              tqdm=tqdm.tqdm,
              writer=None,
              hps=hps,
              save_model_path="../models/test",
              number=0,
              iter_max=2,
              status="new"
              )
    print("train_cnn code completed. Onto adversarial_rejacc_cnn code...")
    ic_adv_, img_p_adv = adversarial_rejacc_cnn(model=model,
                                               T=x_train.shape[1]+4,
                                               formula=formula,
                                               formula_input_func=formula_input_func,
                                               device=device,
                                               hps=hps,
                                               save_model_path="../models/test",
                                               number=0,
                                               lower=lower,
                                               upper=upper,
                                               adv_n_samples=128)
    n_adv = img_p_adv.shape[0]
    random_n = torch.randperm(n_adv)[:n_train]
    ic_adv_ = ic_adv_[random_n]
    img_p_adv = img_p_adv[random_n]

    ic_train_ = initial_conditions(n_train, lower, upper)
    ic_eval_ = initial_conditions(n_eval, lower, upper)

    ic_train = model.standardize_x(ic_train_.to(device).float())
    ic_eval = model.standardize_x(ic_eval_.to(device).float())
    ic_adv = model.standardize_x(ic_adv_.to(device).float())



    ic_train = InitialConditionDatasetCNN(ic_train, ic_adv, img_p_adv)
    ic_eval = InitialConditionDatasetCNN(ic_eval)

    ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=n_train//16, shuffle=True)
    ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=n_eval, shuffle=False)


    train_cnn(model=model,
              train_traj=(x_train, u_train),
              eval_traj=(x_eval, u_eval),
              imgs=(imgs_train, imgs_eval),
              tls=(tls_train, tls_eval),
              centers=(centers_train, centers_eval),
              formula=formula,
              formula_input_func=formula_input_func,
              train_loader=ic_trainloader,
              eval_loader=ic_evalloader,
              device=device,
              tqdm=tqdm.tqdm,
              writer=None,
              hps=hps,
              save_model_path="../models/test",
              number=0,
              iter_max=2,
              status="new"
              )

    print("adversarial_rejacc_cnn code completed.")

    write_log(log_dir, "Testing: done!")

elif args.mode == "train":

    ic_train_ = initial_conditions(args.trainset_size, lower, upper)
    ic_eval_ = initial_conditions(args.evalset_size, lower, upper)


    # standardized data
    x_train = model.standardize_x(x_train_.to(device).float())
    u_train = model.standardize_u(u_train_.to(device).float())

    x_eval = model.standardize_x(x_eval_.to(device).float())
    u_eval = model.standardize_u(u_eval_.to(device).float())

    ic_train = model.standardize_x(ic_train_.to(device).float())
    ic_eval = model.standardize_x(ic_eval_.to(device).float())

    ic_train = InitialConditionDatasetCNN(ic_train)
    ic_eval = InitialConditionDatasetCNN(ic_eval)

    ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=args.trainset_size//8, shuffle=True)
    ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=args.evalset_size, shuffle=False)

    train_cnn(model=model,
              train_traj=(x_train, u_train),
              eval_traj=(x_eval, u_eval),
              imgs=(imgs_train, imgs_eval),
              tls=(tls_train, tls_eval),
              centers=(centers_train, centers_eval),
              formula=formula,
              formula_input_func=formula_input_func,
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
                         formula_input_func=formula_input_func,
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
    prompt = input("Adversarial training finished, final comments:")
    write_log(log_dir, "Adversarial: done! {}".format(prompt))



elif args.mode == "adv_training_iteration":

    ic_train_ = initial_conditions(args.trainset_size, lower, upper)
    ic_eval_ = initial_conditions(args.evalset_size, lower, upper)

    x_train = model.standardize_x(x_train_.to(device).float())
    u_train = model.standardize_u(u_train_.to(device).float())

    x_eval = model.standardize_x(x_eval_.to(device).float())
    u_eval = model.standardize_u(u_eval_.to(device).float())

    ic_train = model.standardize_x(ic_train_.to(device).float())
    ic_eval = model.standardize_x(ic_eval_.to(device).float())

    ic_train = InitialConditionDatasetCNN(ic_train)
    ic_eval = InitialConditionDatasetCNN(ic_eval)

    mini_batch = 8

    ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=mini_batch, shuffle=True)
    ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=args.evalset_size, shuffle=False)

    writer = SummaryWriter(log_dir=runs_dir)

    start_idx = 0
    if args.status == "continue":
        start_idx = args.number
    for rep in range(start_idx, start_idx + 3):

        train_cnn(model=model,
                  train_traj=(x_train, u_train),
                  eval_traj=(x_eval, u_eval),
                  imgs=(imgs_train, imgs_eval),
                  tls=(tls_train, tls_eval),
                  centers=(centers_train, centers_eval),
                  formula=formula,
                  formula_input_func=formula_input_func,
                  train_loader=ic_trainloader,
                  eval_loader=ic_evalloader,
                  device=device,
                  tqdm=tqdm.tqdm,
                  writer=writer,
                  hps=hps,
                  save_model_path=model_dir,
                  number=rep,
                  iter_max=args.iter_max * (rep + 1),
                  status=args.status if (rep== 0) else "continue"
                  )

        write_log(log_dir, "Training: {} training phase(s) done".format(rep+1))

        ic_adv_, img_p_adv = adversarial_rejacc_cnn(model=model,
                                                    T=x_train.shape[1]+4,
                                                    formula=formula,
                                                    formula_input_func=formula_input_func,
                                                    device=device,
                                                    writer=writer,
                                                    hps=hps,
                                                    save_model_path=model_dir,
                                                    number=rep,
                                                    lower=lower,
                                                    upper=upper,
                                                    adv_n_samples=args.adv_n_samples
                                                    )

        write_log(log_dir, "Adversarial number={} done".format(rep))

        trainset_size = args.trainset_size

        if ic_adv_ is not None:
            n_adv = img_p_adv.shape[0]
            write_log(log_dir, "{} adversarial samples found".format(n_adv))
            if n_adv < (mini_batch // 2):
                ic_adv = None
                img_p_adv = None
                print("Too few adversarial samples")

            else:
                ic_adv = model.standardize_x(ic_adv_.to(device).float())
        else:
            ic_adv = None
            write_log(log_dir, "{} adversarial samples found".format(0))



        ic_train_ = initial_conditions(n_adv, lower, upper)
        ic_eval_ = initial_conditions(args.evalset_size, lower, upper)

        ic_train = model.standardize_x(ic_train_.to(device).float())
        ic_eval = model.standardize_x(ic_eval_.to(device).float())

        ic_train = InitialConditionDatasetCNN(ic_train, ic_adv, img_p_adv)
        ic_eval = InitialConditionDatasetCNN(ic_eval)

        ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=mini_batch//2, shuffle=True)
        ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=args.evalset_size, shuffle=False)


        
elif args.mode == "adv_training_iteration_rapid":

    ic_train_ = initial_conditions(args.trainset_size, lower, upper)
    ic_eval_ = initial_conditions(args.evalset_size, lower, upper)

    x_train = model.standardize_x(x_train_.to(device).float())
    u_train = model.standardize_u(u_train_.to(device).float())

    x_eval = model.standardize_x(x_eval_.to(device).float())
    u_eval = model.standardize_u(u_eval_.to(device).float())

    ic_train = model.standardize_x(ic_train_.to(device).float())
    ic_eval = model.standardize_x(ic_eval_.to(device).float())

    ic_train = InitialConditionDatasetCNN(ic_train)
    ic_eval = InitialConditionDatasetCNN(ic_eval)

    mini_batch = 8

    ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=mini_batch, shuffle=True)
    ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=args.evalset_size, shuffle=False)

    writer = SummaryWriter(log_dir=runs_dir)

    # # do complete training round to get a decent solution
    # train_cnn(model=model,
    #               train_traj=(x_train, u_train),
    #               eval_traj=(x_eval, u_eval),
    #               imgs=(imgs_train, imgs_eval),
    #               tls=(tls_train, tls_eval),
    #               centers=(centers_train, centers_eval),
    #               formula=formula,
    #               formula_input_func=formula_input_func,
    #               train_loader=ic_trainloader,
    #               eval_loader=ic_evalloader,
    #               device=device,
    #               tqdm=tqdm.tqdm,
    #               writer=writer,
    #               hps=hps,
    #               save_model_path=model_dir,
    #               number=0,
    #               iter_max=args.iter_max,
    #               status=args.status
    #               )

    write_log(log_dir, "First training phase done:")

    hps = hyperparameters(weight_decay=0.05,
              learning_rate=0.1,
              teacher_training=lambda ep: 1.0,
              weight_ctrl=args.weight_ctrl,
              weight_recon=args.weight_recon,
              weight_stl=lambda ep: args.stl_max,
              stl_scale=lambda ep: args.scale_max,
              adv_stl_scale=1.0,
              alpha=0.001,
              img_bs=4,
              expert_mini_bs=args.expert_mini_bs)

    mini_iter_max = 5
    for rep in range(3):

        train_cnn(model=model,
                  train_traj=(x_train, u_train),
                  eval_traj=(x_eval, u_eval),
                  imgs=(imgs_train, imgs_eval),
                  tls=(tls_train, tls_eval),
                  centers=(centers_train, centers_eval),
                  formula=formula,
                  formula_input_func=formula_input_func,
                  train_loader=ic_trainloader,
                  eval_loader=ic_evalloader,
                  device=device,
                  tqdm=tqdm.tqdm,
                  writer=writer,
                  hps=hps,
                  save_model_path=model_dir,
                  number=rep+1,
                  iter_max=args.iter_max * args.number + mini_iter_max * (rep + 1),
                  status="continue"
                  )


        ic_adv_, img_p_adv = adversarial_rejacc_cnn(model=model,
                                                    T=x_train.shape[1]+4,
                                                    formula=formula,
                                                    formula_input_func=formula_input_func,
                                                    device=device,
                                                    writer=writer,
                                                    hps=hps,
                                                    save_model_path=model_dir,
                                                    number=rep,
                                                    lower=lower,
                                                    upper=upper,
                                                    adv_n_samples=args.adv_n_samples
                                                )
        trainset_size = args.trainset_size

        if ic_adv_ is not None:
            n_adv = img_p_adv.shape[0]
            write_log(log_dir, "{} adversarial samples found".format(n_adv))
            if n_adv < (mini_batch // 2):
                ic_adv = None
                img_p_adv = None
                print("Too few adversarial samples")

            else:
                ic_adv = model.standardize_x(ic_adv_.to(device).float())
        else:
            ic_adv = None
            write_log(log_dir, "{} adversarial samples found".format(0))



        ic_train_ = initial_conditions(n_adv, lower, upper)
        ic_eval_ = initial_conditions(args.evalset_size, lower, upper)

        ic_train = model.standardize_x(ic_train_.to(device).float())
        ic_eval = model.standardize_x(ic_eval_.to(device).float())

        ic_train = InitialConditionDatasetCNN(ic_train, ic_adv, img_p_adv)
        ic_eval = InitialConditionDatasetCNN(ic_eval)

        ic_trainloader = torch.utils.data.DataLoader(ic_train, batch_size=mini_batch//2, shuffle=True)
        ic_evalloader = torch.utils.data.DataLoader(ic_eval, batch_size=args.evalset_size, shuffle=False)


else:
    raise NameError("args.mode undefined")
