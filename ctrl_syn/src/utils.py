import sys
import matplotlib.pyplot as plt
import numpy as np
import torch



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
