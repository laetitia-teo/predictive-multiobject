"""
A small utility file for the dataset.
"""
import h5py
import os
import os.path as op
import re

import numpy as np
import torch

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def load_hdf5(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    data = []

    with h5py.File(fname, 'r') as f:
        for i, grp in enumerate(f.keys()):
            data.append({})
            for key in f[grp].keys():
                data[i][key] = f[grp][key][:]
    return data

def load_hdf5(fname, device=torch.device('cpu')):
    data = []

    with h5py.File(fname, 'r') as f:
        for grp in f.keys():
            data.append(torch.tensor(f[grp][()], device=device))

    return data

def transpose_transform(np_array):
    a = np.transpose(np_array, (2, 0, 1))
    return a.astype(np.float32)

class ImageDs(Dataset):
    """
    Image Dataset. Each sample is a whole sequence.

    We assume everything fits into memory.
    """
    def __init__(self, path, gpu=False, seq_limit=None, max_samples=8,
                 load_prefix=None):
        
        self.seq_limit = seq_limit
        self.max_samples = max_samples

        if gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        if load_prefix is None:
            self.load_data(path)
        # elif isinstance(load_prefix, list):
        else:
            self.load_data_only_prefix(path, load_prefix)

    def load_data(self, path, mode="new"):
        # load data
        # mode can be "new" or "append"
        l = os.listdir(path)
        
        def indices(s):
            r = re.search(r'^sample([0-9]+)frame([0-9]+).png$', s)
            if r is not None:
                return int(r[1]), int(r[2])
            else:
                return -1, -1

        # get nb of datapoints and sequence size
        if mode == "new":
            self.N_samples = max(indices(s)[0] for s in l) + 1
        elif mode == "add":
            self.N_samples += max(indices(s)[0] for s in l) + 1

        self.T = max(indices(s)[1] for s in l) + 1

        if self.seq_limit is not None:
            if mode == "new":
                self.T = min(self.seq_limit, self.T)
            elif mode == "add":
                self.T = min(self.seq_limit, self.T)
        
        if self.max_samples is not None:
            self.N_samples = min(self.max_samples, self.N_samples)

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

    def load_data_only_prefix(self, path, prefix, mode="new"):
        """
        Loads only the data sample beginning with prefix.
        """

        l = os.listdir(path)

        def indices(s):
            r = re.search(rf"^{prefix}frame([0-9]+).png", s)
            if r is not None:
                return int(r[1])
            else:
                return -1

        if mode == "new":
            self.N_samples = 1
        elif mode == "add":
            self.N_samples += 1

        self.T = max(indices(s) for s in l) + 1
        if self.seq_limit is not None:
            self.T = min(self.seq_limit, self.T)
        
        # get image dims
        img = Image.open(op.join(path, l[0]))
        img = np.array(img).astype(np.float32)
        self.H, self.W, self.C = img.shape

        t = torch.zeros(self.N_samples, self.T, self.C, self.H, self.W)
        
        print('loading dataset...')
        for i in range(self.T):
            img = Image.open(op.join(path, f"{prefix}frame{i}.png"))
            img = np.array(img).astype(np.float32)

            img = (img - 127.5) / 127.5
            img = torch.from_numpy(img).permute(2, 0, 1)

            t[0, i] = img
        print('done')

        if mode == "new":
            self.data = t.to(self.device)
        elif mode == "add":
            self.data = torch.cat([self.data, t.to(self.device)], 0)

    def __len__(self):
        return self.N_samples

    def __getitem__(self, i):
        return self.data[i]

class TransitionDataset(Dataset):
    """
    Dataset class for simple transitions.

    # TODO: deprecated
    """
    def __init__(self, fname):
        self.data = load_hdf5(fname)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.data)):
            num_steps = len(self.data[ep])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = transpose_transform(self.data[ep][step])
        next_obs = transpose_transform(self.data[ep][step])

        return obs, next_obs

class SequenceDataset(Dataset):
    """
    Dataset class for containing whole sequences.
    """
    def __init__(self, fname):
        super().__init__()
        self.data = load_hdf5(fname)

    def transform(self, img):
        img = img.permute(0, 3, 1, 2)
        return (img.float() - 127.5) / 127.5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

if __name__ == '__main__':
    fname = '../data/three_body_physics.hdf5'
    # tds = TransitionDataset(fname)
    sds = SequenceDataset(fname)
    None
