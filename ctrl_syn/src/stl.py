import sys
sys.path.append('../../../stlcg_karen/src')
sys.path.append('../../expert_demo_ros/src/utils')
sys.path.append('../')
import stlcg
import matplotlib.pyplot as plt

import numpy as np
import torch
from environment import *

stl_traj = x_train_.permute([1,0,2]).flip(1)

in_end_goal = inside_circle(env.final)
stop_in_end_goal = in_end_goal & (stlcg.Expression('speed')  < 0.5)
end_goal = stlcg.Eventually(subformula=stop_in_end_goal)
coverage = stlcg.Eventually(subformula=(always_inside_circle(env.covers[0], interval=[0,10]) & (stlcg.Expression('speed')  < 1.0)))
avoid_obs = always_outside_circle(env.obs[0])

stl_formula = stlcg.Until(subformula1=coverage, subformula2=end_goal) & avoid_obs



def inside_circle(cover):
    return stlcg.Expression('d2_to_coverage') < cover.radius**2

def always_inside_circle(cover, interval=[0,5]):
    return stlcg.Always(subformula=inside_circle(cover), interval=interval)

def outside_circle(circle):
    return stlcg.Negation(subformula=inside_circle(circle))

def always_outside_circle(circle, interval=None):
    return stlcg.Always(subformula=outside_circle(circle), interval=interval)

def circle_input(signal, cover, device='cpu', backwards=False):
    if not backwards:
        signal = signal.flip(1)
    return (signal[:,:,:2].to(device) - torch.tensor(cover.center).to(device)).pow(2).sum(-1, keepdim=True)

def speed_input(signal, device='cpu', backwards=False):
    if not backwards:
        signal = signal.flip(1)
    return signal[:,:,3:4].to(device)

def get_formula_input(signal, cover, obs, goal, device, backwards=False):
    # (coverage until goal) & avoid obs
    # coverage = be inside circle and slow down
    # goal = be inside circle and stop
    coverage_input = circle_input(signal, cover, device, backwards)
    avoid_obs_input = circle_input(signal, obs, device, backwards)
    goal_input = circle_input(signal, goal, device, backwards)
    speed_input_ = speed_input(signal, device, backwards)
    return (((coverage_input, speed_input_), (goal_input, speed_input_)), avoid_obs_input)
