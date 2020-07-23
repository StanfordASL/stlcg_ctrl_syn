import matplotlib
import numpy as np
import matplotlib.pyplot as plt


class CoverageEnv:
    def __init__(self, params):
        self.obs = params["obstacles"]
        self.covers = params["covers"]
        self.initial = params["initial"]
        self.final = params["final"]

    def draw2D(self, dims=[0, 1], kwargs={"initial": {}, "final": {}, "covers": {}, "obs": {} }):
        self.initial.draw2D(dims, **kwargs["initial"])
        self.final.draw2D(dims, **kwargs["final"])
        for covs in self.covers:
            covs.draw2D(dims, **kwargs["covers"])
        for obs in self.obs:
            obs.draw2D(dims, **kwargs["obs"])
            
        

class Box:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        
    def draw2D(self, dims=[0,1], fill=False, **kwargs):
        x, y = dims
        lower = self.lower
        upper = self.upper
        x_corners = [lower[x], upper[x], upper[x], lower[x], lower[x]]
        y_corners = [lower[y], lower[y], upper[y], upper[y], lower[y]]
        if not fill:
            plt.plot(x_corners, y_corners, **kwargs)
        else:
            plt.fill(x_corners, y_corners, **kwargs)

            
class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def draw2D(self, dims=[0,1], fill=False, **kwargs):
        x, y = dims
        center = [self.center[x], self.center[y]]
        th = np.arange(-np.pi, np.pi+0.1, 0.05)
        x = center[0] + self.radius * np.cos(th)
        y = center[1] + self.radius * np.sin(th)
        if not fill:
            plt.plot(x, y, **kwargs)
        else:
            plt.fill(x, y, **kwargs)
        

if __name__ == "__main__":
    params = {  "covers": [Box([0., 0.6],[0.3, 0.8]), Box([0.6, 0.2],[1.0, 0.4])],
                "obstacles": [Circle([0.7, 0.7], 0.15)],
                "initial": Box([-0.1, -0.1],[0.1, 0.1]),
                "final": Box([0.9, 0.9],[1.1, 1.1])
           }
    draw_params = {"initial": {"color": "lightskyblue", "fill": True, "alpha": 0.5}, "final": {"color": "coral", "fill": True, "alpha": 0.5}, "covers": {"color": "black", "fill": False}, "obs": {"color": "red", "fill": True, "alpha": 0.5} }
    cov_env = CoverageEnv(params)
    plt.figure(figsize=(10,10))
    cov_env.draw2D(kwargs=draw_params)
    plt.axis("equal")
    plt.show()