import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
import sys
sys.path.append('../ctrl_syn')
from torch_interpolations.torch_interpolations.multilinear import RegularGridInterpolator
import IPython

value = sio.loadmat('stlhj/coverage_DoubleInt_test/value.mat');
deriv_value = sio.loadmat('stlhj/coverage_DoubleInt_test/deriv_value.mat');
V = np.flip(value['data'], 4).copy();
g = [s[0].flatten() for s in sio.loadmat('stlhj/coverage_DoubleInt_test/grid.mat')['grid']];

values = torch.tensor(V).float()
dV = [torch.tensor(deriv_value['derivC'][i][0]).float() for i in range(5)];
points = [torch.from_numpy(g[i].flatten()).float() for i in range(4)] + [torch.from_numpy(np.arange(0,1.05, 0.05)).float()]
value_interp = RegularGridInterpolator(points, values)
deriv_interp = [RegularGridInterpolator(points, dV[i]) for i in range(5)]


class DynSys(object):
    def __init__(self, uMin, uMax, dMin, dMax):
        self.x = None
        self.uMin = uMin
        self.uMax = uMax
        self.dMin = dMin
        self.dMax = dMax

    def dynamics(self, x, u, d, dt):
        raise 

    def optCtrl(self, t, x, deriv, uMode):
        raise NotImplementedError

    def optDstb(self, t, x, deriv, dMode):
        raise NotImplementedError

    def updateState(self, u, dtSmall, x):
        raise NotImplementedError



class DoubleIntegrator2D(DynSys):
    def __init__(self, uMin, uMax):
        super().__init__(uMin, uMax, None, None)
        self.A = np.eye(4)
        self.B = np.zeros([4,2])

    def dynamics(self, x, u, d, dt):
        if type(x) == list:
            x = np.array(x)
        if type(u) == list:
            u = np.array(u)
        # if type(d) == list:
        #     d = np.array(d)
        self.A[0,1] = dt
        self.A[2,3] = dt
        self.B[1,0] = dt
        self.B[3,1] = dt

        return self.A @ x + self.B @ u

    def optCtrl(self, t, x, deriv, uMode):
        uOpt = np.zeros(2)
        if uMode == "max":
            uOpt[0] = (deriv[1] >= 0) * self.uMax[0] + (deriv[1] < 0) * self.uMin[0]
            uOpt[1] = (deriv[3] >= 0) * self.uMax[1] + (deriv[3] < 0) * self.uMin[1]
        elif uMode == "min":
            uOpt[0] = (deriv[1] <= 0) * self.uMax[0] + (deriv[1] > 0) * self.uMin[0]
            uOpt[1] = (deriv[3] <= 0) * self.uMax[1] + (deriv[3] > 0) * self.uMin[1]
        return uOpt

    def updateState(self, u, dtSmall, x):
        self.x = self.dynamics(x, u, None, dtSmall)
        return self.x


def proj(g, x, value_interp, showDims, hideDims, tau, tEarliest):
    sds = np.meshgrid(g[showDims[0]], g[showDims[1]])
    hds = [np.ones_like(sds[0]) * x[hd] for hd in hideDims]
    ts = np.ones_like(sds[0]) * tau[3]
    p = []
    for i in range(len(g)):
        if i in showDims:
            p.append(torch.from_numpy(sds.pop(0)).float())
        else:
            p.append(torch.from_numpy(hds.pop(0)).float())
    p.append(torch.from_numpy(ts).float())
    return np.meshgrid(g[showDims[0]], g[showDims[1]]), value_interp(p)

def compute_value(p, value_interp):
    return value_interp(p).squeeze().numpy().item()

def compute_gradients(p, deriv_interp):
    return [di(p).squeeze().numpy().item() for di in deriv_interp]

def find_earliest_BRS_ind(g, data, x, tau, upper, lower):
    clns = [len(g)] + [i for i in range(len(g))]
    small = 1E-4
    while upper > lower:
        tEarliest = int(np.ceil((upper + lower)/2))
        p = list(torch.tensor(x).split(1, dim=-1)) + [torch.tensor([tau[tEarliest]])]
        valueX = value_interp(p)
        if valueX < small:
            lower = tEarliest
        else:
            upper = tEarliest - 1

    return upper

def computeOptTraj(g, data, tau, dynSys, projDim, **kwargs):
    # Default parameters
    uMode = 'min';
    visualize = False;
    subSamples = 4;
    keys = kwargs.keys()

    if 'uMode' in keys:
        uMode = kwargs['uMode']


    # Visualization
    if ('visualize' in keys) & (kwargs["visualize"] == True):
        visualize = kwargs['visualize']


    if type(projDim) == list:
        projDim = np.array(projDim)

    showDims = np.where(projDim)[0]
    hideDims = np.where(1 - projDim)[0];

    # if 'fig' in keys:
    #     f, ax = kwargs['fig']
    # else:
    #     f, ax = plt.subplots(figsize=(10,10))


    if 'subSamples' in keys:
        subSamples = kwargs['subSamples']

    clns = [len(g)] + [i for i in range(len(g))]


    if any(np.diff(tau)) < 0:
        raise Exception('Time stamps must be in ascending order!')



    # Time parameters
    it = 1;
    tauLength = len(tau);
    dtSmall = (tau[1] - tau[0]) / subSamples

    # Initialize trajectory
    traj = []
    values = []
    tEarliestList = []
    traj.append(dynSys.x);
    tEarliest = 0;

    obs = plt.Circle([-0.25, -0.25], 0.15, color='red')
    goal = plt.Circle([0.05, 0.05], 0.15, color='green')


    while it <= tauLength-1:
        # Determine the earliest time that the current state is in the reachable set
        # Binary search
        upper = tauLength - 1;
        lower = tEarliest;
        tEarliest = find_earliest_BRS_ind(g, data, dynSys.x, tau, upper, lower);
        tEarliestList.append(tEarliest)
        # BRS at current time
        BRS_at_t = np.transpose(data, clns)[tEarliest]
        p = list(torch.tensor(dynSys.x).split(1, dim=-1)) + [torch.tensor([tau[tEarliest]])]
        values.append(compute_value(p, value_interp))
        # Visualize BRS corresponding to current trajectory point
        if visualize:
            f, ax = plt.subplots(figsize=(10,10))
            obs = plt.Circle([-0.25, -0.25], 0.15, color='red')
            goal = plt.Circle([0.05, 0.05], 0.15, color='green')

            traj_flatten = np.stack(traj)
            ax.plot(traj_flatten[:it, showDims[0]], traj_flatten[:it, showDims[1]], 'k')
            # breakpoint()
            # ax.scatter(traj_flatten[:it, showDims[0]], traj_flatten[:it, showDims[1]], 'k')
            g2D, data2D = proj(g, traj[-1], value_interp, showDims, hideDims, tau, tEarliest)
            ax.contour(*g2D, data2D, np.arange(0,5,0.5))
            ax.add_artist(obs)
            ax.add_artist(goal)
            ax.set_title('t = %.3f; tEarliest = %.3f'%(tau[it], tau[tEarliest]))

            if 'fig_filename' in keys:
              f.savefig(kwargs['fig_filename'] + "_{:02d}.png".format(it))



        if tEarliest == tauLength-1:
        # Trajectory has entered the target
            break

        # Update trajectory
        dVt = compute_gradients(p, deriv_interp)    # [∇V , dV/dt]
        for j in range(subSamples):
            u = dynSys.optCtrl(tau[tEarliest], dynSys.x, dVt, uMode)
            dynSys.updateState(u, dtSmall, dynSys.x)

        # Record new point on nominal trajectory
        it = it + 1;
        traj.append(dynSys.x)
    # Delete unused indices
    traj_tau = tau[:it];

    return traj, traj_tau, values, tEarliest


if __name__ == '__main__':
    kwargs = {'uMode': 'min', 'visualize': True}
    projDim = [1,0,1,0]
    data = V
    tau = np.arange(0, 1.05, 0.05)
    uMin = [-2,-2]
    uMax = [2,2]
    dynSys = DoubleIntegrator2D(uMin, uMax)
    dynSys.x = np.array([-0.45, 0.5, -0.25, 0.0])
    traj, traj_tau, values, tEarliest = computeOptTraj(g, data, tau, dynSys, projDim, uMode="min", visualize=True,fig_filename='stlhj/test')
    print(values)
