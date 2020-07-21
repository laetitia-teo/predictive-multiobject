"""
A small utility file for the dataset.
"""
import os
import os.path as op
import re

import numpy as np
import torch

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageDs(Dataset):
    """
    Image Dataset. Not episodic (one single episode).

    We assume everything fits into memory.
    """
    def __init__(self, path, gpu=True, dt=1):
        
        self.path = path
        self.dt = dt

        if gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # load data
        l = os.listdir(self.path)
        
        def index(s):
            r = re.search(r'^frame([0-9]+).png$', s)
            if r is not None:
                return int(r[1])
            else:
                return -1

        # get nb of datapoints
        self.N = max(index(s) for s in l)
        # get image dims
        img = Image.open(op.join(path, l[0]))
        img = np.array(img).astype(np.float32)
        self.H, self.W, self.C = img.shape

        t = torch.zeros(self.N, self.C, self.H, self.W)
        
        print('loading dataset...')
        for i in tqdm(range(self.N)):
            img = Image.open(op.join(path, f"frame{i}.png"))
            img = np.array(img).astype(np.float32)

            img = (img - 127.5) / 127.5
            img = torch.from_numpy(img).permute(2, 0, 1)

            t[i] = img
        print('done')

        self.data = t.to(self.device)

    def __len__(self):
        return self.N - self.dt

    def __getitem__(self, i):
        st = self.data[i]
        stn = self.data[i + self.dt]
        p = np.random.choice(self.N) # TODO: exclude i + dt ?
        stp = self.data[p]

        return st, stn, stp