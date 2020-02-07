from torch.utils.data import Dataset
import torch
import pandas as pd
import trimesh
import scipy.ndimage as nd
import numpy as np
import gc




class ObjectDataset(Dataset): 
    
    def __init__(self, csv_file, side_len=32):
        self.df = pd.read_csv(csv_file, index_col=0)
        self.side_len = side_len
    
    def __getitem__(self, index):
        item = self.df.loc[[index]]
        path = np.array(item.iloc[0])[0]
        return self.__getVolume__(path)
    
    def __getVolume__(self, path):
        mesh = trimesh.load_mesh(path)
        extend = max(mesh.extents)
        volume = mesh.voxelized(round(extend/self.side_len, 3)).matrix
        suma = max(volume.shape) - np.array(volume.shape)
        volume = np.pad(volume, max(suma), mode="constant")
        (x, y, z) = map(float, volume.shape)
        volume = nd.zoom(volume.astype(float), 
                         (self.side_len/x, self.side_len/y, self.side_len/z),
                         order=1, 
                         mode='constant',
                         cval=0
                        )
        volume[np.nonzero(volume)] = 1.0
        return torch.FloatTensor(volume)



    def __len__(self):
        return self.df.shape[0]
    