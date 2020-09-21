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
        else:
            self.load_data_only_prefix(path, load_prefix)

    def load_data(self, path):
        # load data
        l = os.listdir(path)
        
        def indices(s):
            r = re.search(r'^sample([0-9]+)frame([0-9]+).png$', s)
            if r is not None:
                return int(r[1]), int(r[2])
            else:
                return -1, -1

        # get nb of datapoints and sequence size
        self.N_samples = max(indices(s)[0] for s in l) + 1
        self.T = max(indices(s)[1] for s in l) + 1
        if self.seq_limit is not None:
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

    def load_data_only_prefix(self, path, prefix):
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

        self.N_samples = 1
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

        self.data = t.to(self.device)

    def __len__(self):
        return self.N_samples

    def __getitem__(self, i):
        return self.data[i]

class ChunkImageDS(Dataset):
    """
    Chunk version of ImageDs: we define an additional L param that controls the
    length of the output sequences.

    Cuts the original squences of length self.T in chunks of length L. Returns
    each chunk only once.
    """
    def __init__(self, path, L, gpu=False):
        super().__init__(path, gpu)

        self.L = L

    def __len__(self):
        return self.N_samples * (self.T // self.L)

    def __getitem__(self, i):
        n_chunks = self.T // self.L
        k, r = i // n_chunks, i % n_chunks
        seq = self.data[k]
        return seq[r:r+self.L]

class ChunkImageDS2(Dataset):
    """
    Chunk version of ImageDs: we define an additional L param that controls the
    length of the output sequences.

    Cuts the original squences of length self.T in chunks of length L. The 
    chunks beginning indices span all sequence elements.

    TODO: finish this
    """
    def __init__(self, path, L, gpu=False):
        super().__init__(path, gpu)

        self.L = L

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError