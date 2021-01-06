import sys
sys.path.append('../../../stlcg_karen/src')
sys.path.append('../../expert_demo_ros/src/utils')
sys.path.append('../')
import stlcg
import matplotlib.pyplot as plt

import numpy as np
import torch
from environment import *



def inside_circle(cover, name):
    return stlcg.Expression(name) < cover.radius**2

def always_inside_circle(cover, name, interval=[0,5]):
    return stlcg.Always(subformula=inside_circle(cover, name), interval=interval)

def outside_circle(circle, name):
    return stlcg.Negation(subformula=inside_circle(circle, name))

def always_outside_circle(circle, name, interval=None):
    return stlcg.Always(subformula=outside_circle(circle, name), interval=interval)

def circle_input(signal, cover, device='cpu', backwards=False, time_dim=1):
    if not backwards:
        signal = signal.flip(time_dim)
    if isinstance(cover.center, torch.Tensor):
        center = cover.center
    else:
        center = torch.Tensor(cover.center)
    return (signal[:,:,:2].to(device) - center.to(device)).pow(2).sum(-1, keepdim=True).float()

def speed_input(signal, device='cpu', backwards=False):
    if not backwards:
        signal = signal.flip(1)
    return signal[:,:,3:4].to(device).float()

def get_formula_input(signal, cover, obs, goal, device, backwards=False):
    # (coverage until goal) & avoid obs
    # coverage = be inside circle and slow down
    # goal = be inside circle and stop
    coverage_input = circle_input(signal, cover, device, backwards)
    avoid_obs_input = circle_input(signal, obs, device, backwards)
    goal_input = circle_input(signal, goal, device, backwards)
    speed_input_ = speed_input(signal, device, backwards)
    return (((coverage_input, speed_input_), (goal_input, speed_input_)), avoid_obs_input)
