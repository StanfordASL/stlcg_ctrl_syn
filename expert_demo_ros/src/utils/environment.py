import matplotlib
import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, params):
        self.obs = params["obstacles"]
        self.covers = params["covers"]
        self.initial = params["initial"]
        self.final = params["final"]

    def draw2D(self, dims=[0, 1], ax=None, kwargs={"initial": {}, "final": {}, "covers": {}, "obs": {} }):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        _, ax = self.initial.draw2D(dims, ax=ax, **kwargs["initial"])
        self.final.draw2D(dims, ax=ax, **kwargs["final"])
        for covs in self.covers:
            _, ax = covs.draw2D(dims, ax=ax, **kwargs["covers"])
        for obs in self.obs:
            _, ax = obs.draw2D(dims, ax=ax, **kwargs["obs"])

        return fig, ax
            
        

class Box:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        
    def center(self):
        return [(l + u)/2 for (l,u) in zip(self.lower, self.upper)]

    def draw2D(self, dims=[0,1], fill=False, ax=None, **kwargs):
        x, y = dims
        lower = self.lower
        upper = self.upper
        x_corners = [lower[x], upper[x], upper[x], lower[x], lower[x]]
        y_corners = [lower[y], lower[y], upper[y], upper[y], lower[y]]
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        if not fill:
            ax.plot(x_corners, y_corners, **kwargs)
        else:
            ax.fill(x_corners, y_corners, **kwargs)


        return fig, ax
            
class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def center(self):
        return self.center
        
    def draw2D(self, dims=[0,1], ax=None, fill=False, **kwargs):
        x, y = dims
        center = [self.center[x], self.center[y]]
        th = np.arange(-np.pi, np.pi+0.1, 0.05)
        x = center[0] + self.radius * np.cos(th)
        y = center[1] + self.radius * np.sin(th)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        if not fill:
            ax.plot(x, y, **kwargs)
        else:
            ax.fill(x, y, **kwargs)
        
        return fig, ax
            
if __name__ == "__main__":
    params = {  "covers": [Box([0., 0.6],[0.3, 0.8]), Box([0.6, 0.2],[1.0, 0.4])],
                "obstacles": [Circle([0.7, 0.7], 0.15)],
                "initial": Box([-0.1, -0.1],[0.1, 0.1]),
                "final": Box([0.9, 0.9],[1.1, 1.1])
           }
    draw_params = {"initial": {"color": "lightskyblue", "fill": True, "alpha": 0.5}, "final": {"color": "coral", "fill": True, "alpha": 0.5}, "covers": {"color": "black", "fill": False}, "obs": {"color": "red", "fill": True, "alpha": 0.5} }
    env = Environment(params)
    fig, ax = plt.subplots(figsize=(10,10))
    _, ax = env.draw2D(kwargs=draw_params)
    ax.axis("equal")
    plt.show()