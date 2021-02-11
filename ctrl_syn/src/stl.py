import sys
sys.path.append('../../../stlcg_karen/src')
sys.path.append('../../expert_demo_ros/src/utils')
sys.path.append('../')
import stlcg
import matplotlib.pyplot as plt

import numpy as np
import torch
from environment import *
import IPython


def inside_circle(cover, name):
    return stlcg.Expression(name) < cover.radius

def always_inside_circle(cover, name, interval=[0,5]):
    return stlcg.Always(subformula=inside_circle(cover, name), interval=interval)

def outside_circle(circle, name):
    return stlcg.Negation(subformula=inside_circle(circle, name))

def always_outside_circle(circle, name, interval=None):
    return stlcg.Always(subformula=outside_circle(circle, name), interval=interval)

def circle_input(signal, p, device='cpu', backwards=False, time_dim=1, LF=0.5, LR=0.7, W=1.2/2):
    if not backwards:
        signal = signal.flip(time_dim)
    if isinstance(p.center, torch.Tensor):
        center = p.center.to(device).float()
    else:
        center = torch.Tensor(p.center).to(device).float()
    return distance_rec_point(signal, center, LF=LF, LR=LR, W=W)

def rec_input(signal, box, device="cpu", backwards=False, time_dim=1):
    if not backwards:
        signal = signal.flip(time_dim)
    return distance_rec_rec(signal, box, device=device)

def wall_input(signal, wall, device='cpu', backwards=False, time_dim=1, LF=0.5, LR=0.7, W=1.2/2):
    if not backwards:
        signal = signal.flip(time_dim)
    if isinstance(wall[0], torch.Tensor):
        wall[0] = wall[0].to(device).float()
    else:
        wall[0] = torch.Tensor(wall[0]).to(device).float()
    if isinstance(wall[1], torch.Tensor):
        wall[1] = wall[1].to(device).float()
    else:
        wall[1] = torch.Tensor(wall[1]).to(device).float()
    return distance_rec_wall(signal, wall, LF=LF, LR=LR, W=W)
# def circle_input(signal, cover, device='cpu', backwards=False, time_dim=1):
#     if not backwards:
#         signal = signal.flip(time_dim)
#     if isinstance(cover.center, torch.Tensor):
#         center = cover.center.to(device)
#     else:
#         center = torch.Tensor(cover.center).to(device)
#     return (signal[:,:,:2].to(device) - center).pow(2).sum(-1, keepdim=True).float()

def speed_input(signal, device='cpu', backwards=False):
    if not backwards:
        signal = signal.flip(1)
    return signal[:,:,3:4].to(device).float()

def get_formula_input_coverage(signal, env, device, backwards=False):
# def get_formula_input_coverage(signal, cover, obs, goal, device, backwards=False):
    # (coverage until goal) & avoid obs
    # coverage = be inside circle and slow down
    # goal = be inside circle and stop
    cover = env.covers[0]
    obs = env.obs[0]
    goal = env.final
    
    coverage_input = circle_input(signal, cover, device, backwards)
    avoid_obs_input = circle_input(signal, obs, device, backwards)
    goal_input = circle_input(signal, goal, device, backwards)
    speed_input_ = speed_input(signal, device, backwards)
    return (((coverage_input, speed_input_), (goal_input, speed_input_)), avoid_obs_input)


def get_formula_input_drive(signal, env, device, backwards=False):
    left_wall = [torch.tensor([1, 0]).view(1, 1, 2).float(), torch.tensor([env.obs[0].upper[0], 0]).view(1, 1, 2).float()]
    right_wall = [torch.tensor([-1, 0]).view(1, 1, 2).float(), torch.tensor([env.obs[1].lower[0], 0]).view(1, 1, 2).float()]

    distance_to_left_wall = wall_input(signal, left_wall, device=device, backwards=backwards)
    distance_to_right_wall = wall_input(signal, right_wall, device=device, backwards=backwards)
    distance_to_left_lane_obs = circle_input(signal, env.obs[2], device=device, backwards=backwards)
    distance_to_right_lane_obs = circle_input(signal, env.obs[3], device=device, backwards=backwards)
    distance_to_goal = rec_input(signal, env.final, device=device, backwards=backwards)
    
    avoid_obstacle_input = ((distance_to_left_wall, distance_to_right_wall), (distance_to_left_lane_obs, distance_to_right_lane_obs))
    slow_near_obs_input = ((distance_to_left_lane_obs, distance_to_right_lane_obs), speed_input(signal, device=device, backwards=backwards))
    speed_up_goal_input = (distance_to_goal, speed_input(signal, device=device, backwards=backwards))
    return ((speed_up_goal_input, avoid_obstacle_input), slow_near_obs_input)

def perpendicular_distance(n, p, q):
    ''' 
    n: normal vector of line
    p: point we want to compute the distance of
    q: point on the line.
    '''
#     return (np.dot(n, p) - np.dot(n, q)) / np.sqrt(np.dot(n,n))
    return ((n * p).sum(-1, keepdims=True) - (n * q).sum(-1, keepdims=True))/ (n * n).sum(-1, keepdims=True).sqrt()

def get_rec_edges(traj, LF=0.5, LR=0.7, W=1.2/2):
    # get edges of the car given the state/traj
    if isinstance(traj, torch.Tensor):
        ns = torch.stack([torch.cat([torch.cos(traj[:,:,2:3] + np.pi/2 * i), torch.sin(traj[:,:,2:3] + np.pi/2 * i)], dim=-1) for i in range(4)])    # [4, bs, time, 2]
        ds = [LF, W/2, LR, W/2]
        qs = torch.stack([traj[:,:,:2] + di*ni for (di,ni) in zip(ds, ns)])
        # get edges of a Box object
    else:
        rec = traj
        center = torch.tensor([(li+ui)/2 for (li,ui) in zip(rec.lower, rec.upper)]).view(1,1,2).float()    # [1, 1, 2]
        w, h = [ui-li for (li,ui) in zip(rec.lower, rec.upper)]
        ns = torch.tensor(np.array([[np.cos(np.pi/2 + np.pi/2 * i), np.sin(np.pi/2 + np.pi/2 * i)] for i in range(4)])).view(4, 1, 1, 2).float()    # [4, bs, time, 2]
        ds = [h/2, w/2, h/2, w/2]
        qs = torch.stack([center[:,:,:2] + di*ni for (di,ni) in zip(ds, ns)]).float()
    # else:
    #     raise Exception("traj type unknown")
    return ns, qs

def distance_rec_point(traj, p, LF=0.5, LR=0.7, W=1.2/2):
    ns, qs = get_rec_edges(traj, LF=LF, LR=LR, W=W)
    ds = torch.stack([perpendicular_distance(ni, p, qi) for (ni, qi) in zip(ns, qs)])
    return torch.where((ds >=0).sum(0) > 0, ds.relu().pow(2).sum(0).sqrt(), ds.max(0)[0])

def distance_rec_rec(traj, rec, device="cpu"):
    ns, qs = get_rec_edges(rec)
    ds = torch.stack([perpendicular_distance(ni, traj[:,:,:2], qi) for (ni, qi) in zip(ns.to(device), qs.to(device))])
    return -ds.max(0)[0]

def distance_rec_wall(traj, wall, LF=0.5, LR=0.7, W=1.2/2):
    ns, qs = get_rec_edges(traj, LF=LF, LR=LR, W=W)
    corners = torch.stack([qi + di * ns[i-1] for (i ,(qi, di)) in enumerate(zip(qs, [W/2, LF, W/2, LR]))])
    return torch.stack([perpendicular_distance(wall[0], ci, wall[1]) for ci in corners]).min(0)[0]
