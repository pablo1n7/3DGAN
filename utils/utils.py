from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import skimage.measure as sk
import torch.nn as nn

def save_plot_voxels(voxels, path, iteration):
    voxels = voxels.__ge__(0.05)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 5)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(voxels):
        x = sample.nonzero()[:, 0]
        y = sample.nonzero()[:, 2]
        z = sample.nonzero()[:, 1]
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red', s=2.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    plt.savefig(path + '_{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    
    

def cubes2mesh(name, voxels, threshold=0.09, save=False):
    """Voxel Vertices, faces"""
    voxels = np.pad(voxels, 1, mode="constant", constant_values=0)
    v, f, _, _ = sk.marching_cubes_lewiner(voxels)
    sample_mesh = trimesh.Trimesh(v, f)
    sample_mesh.fix_normals()
    if save: 
        sample_mesh.export('./{}.ply'.format(name))
    
    return v, f


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()