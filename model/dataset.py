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
    def __init__(self, path, gpu=False, dt=1):
        
        self.path = path
        self.dt = dt

        if gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # load data
        l = os.listdir(self.path)
        
        def indices(s):
            r = re.search(r'^sample([0-9]+)frame([0-9]+).png$', s)
            if r is not None:
                return int(r[1]), int(r[2])
            else:
                return -1, -1

        # get nb of datapoints and sequence size
        self.N_samples = max(indices(s)[0] for s in l) + 1
        self.T = max(indices(s)[1] for s in l)
        # get image dims
        img = Image.open(op.join(path, l[0]))
        img = np.array(img).astype(np.float32)
        self.H, self.W, self.C = img.shape

        t = torch.zeros(self.N_samples, self.T, self.C, self.H, self.W)
        
        print('loading dataset...')
        for n in tqdm(range(self.N_samples)):
            for i in range(self.T):
                img = Image.open(op.join(path, f"sample{n}frame{i}.png"))
                img = np.array(img).astype(np.float32)

                img = (img - 127.5) / 127.5
                img = torch.from_numpy(img).permute(2, 0, 1)

                t[n, i] = img
        print('done')

        self.data = t.to(self.device)

    def __len__(self):
        return self.N_samples

    def __getitem__(self, i):
        return self.data[i]