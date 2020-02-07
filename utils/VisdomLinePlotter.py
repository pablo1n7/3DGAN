from visdom import Visdom

from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import skimage.measure as sk
from utils.utils import cubes2mesh


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom(port=8090, env=env_name)
        self.env = env_name
        self.plots = {}
        self.scores_window = None
        self.image_window = None

    def plot(self, var_name, split_name, x, y, x_label='Epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel=x_label,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')
            
    def close_window(self, var_name):
        self.viz.close(self.plots[var_name])
        del self.plots[var_name]
        
    def plot_voxels(self, name, voxels, title, savePLY=False):
        v, f = cubes2mesh(name, voxels, save=savePLY)
        self.viz.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))