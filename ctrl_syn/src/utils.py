import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import shutil



def plot_xy_from_tensor(x_train, ax=None):
    xy = x_train.squeeze().detach().numpy()[:,:2]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))
    else:
        fig = None
    ax.plot(xy[:,0], xy[:,1])
    ax.scatter(xy[:,0], xy[:,1])
    return fig, ax

def write_log(fname, msg):
    f = open(fname, "a")
    f.write(msg)
    f.write('\n')
    f.close()

def remove_directory(dir):
    try:
        shutil.rmtree(dir)
    except FileNotFoundError:
        print("%s folder does not exist"%dir)
    sys.exit() 

def make_directory(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        print("%s folder already exists"%dir)


# in_end_goal, in_end_goal_input = in_box_stl(stl_traj, env.final, device)
# stop_in_end_goal, stop_in_end_goal_input = stop_in_box_stl(stl_traj, env.final, device)
# end_goal = stlcg.Eventually(subformula=stlcg.Always(subformula=stop_in_end_box))

# if len(params["covers"]) == 2:
#     in_cov_1, in_cov_1_input = in_box_stl(stl_traj, env.covers[0], device)
#     in_cov_2, in_cov_2_input = in_box_stl(stl_traj, env.covers[1], device)
#     coverage_1 = stlcg.Eventually(subformula=stlcg.Always(subformula=in_cov_1, interval=[0, 10]))
#     coverage_2 = stlcg.Eventually(subformula=stlcg.Always(subformula=in_cov_2, interval=[0, 10]))

# if len(params["covers"]) == 1:
#     in_cov_1, in_cov_1_input = in_box_stl(stl_traj, env.covers[0], device)
#     coverage_1 = stlcg.Eventually(subformula=stlcg.Always(subformula=in_cov_1, interval=[0, 10]))

# if args.type == "goal":
#     formula = obs_avoid & end_goal
#     get_formula_input = get_goal_formula_input
# elif args.type == "coverage":
#     coverage_stl = stlcg.Until(subformula1=coverage_1, subformula2=coverage_2)
#     formula = stlcg.Until(subformula1=coverage_stl, subformula2=end_goal) & obs_avoid
#     get_formula_input = get_coverage_formula_input
# elif args.type == "test":
#     formula = end_goal
#     get_formula_input = get_test_formula_input
# elif args.type == "coverage_test":
#     formula = stlcg.Until(subformula1=coverage_1, subformula2=end_goal) & obs_avoid
#     get_formula_input = get_coverage_test_formula_input
# else:
#     raise NameError(args.type + " is not defined.")