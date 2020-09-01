"""
The module that defines the training loop.

Hyperparamer of the experiments:
    - model components and internal hparams 
        (number of channels,
         number of heads);
    - K (number of slots);
    - image size (INPUT_DIMS);
    - feature size;
    - batch size (BATCH_SIZE);
    - length of sequence
    - number of epochs (N_EPOCHS);
    - learning rate (L_RATE);
    - speed of motion in dataset (?)
"""

# TODO: define experiment id with results, logfiles and saved models
# TODO: Maybe define a process that shows the training curve while the process
# unfolds with subprocess and pygame ?

import os
import re
import datetime
import os.path as op
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader

# TODO use a model dict
from models.models import (CompleteModel_SlotDistance,
                           CompleteModel_SoftMatchingDistance,
                           CompleteModel_HardMatchingDistance,
                           recurrent_apply_contrastive)
from models.dataset import ImageDs

### Constants

N_EPOCHS = 200
BATCH_SIZE = 8
L_RATE = 1e-3
INPUT_DIMS = (100, 100, 3)
K = 2
F_MEM = 2
HIDDEN_DIM = 512
N_HEADS = 1
# EXPE_IDX = 5
BETA = 1. # multiplicative factor for g function
CLIP_GRAD = 1e-3 # ?

DEBUG = False # results go in DEBUG folder
REDO_EXPE = True # wether to redo previous experiment

### Arguments for running training loops

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task',
                    dest='task',
                    default='two_spheres')
parser.add_argument('--N' '--n_epochs',
                    dest='N',
                    default=N_EPOCHS)
parser.add_argument('--B' '--batch_size',
                    dest='bsize',
                    default=BATCH_SIZE)

args = parser.parse_args()

### Logging and file utils

def get_expe_idx():
    # automatically get the index of the experiment
    dirs = os.listdir(op.join("saves", args.task))
    max_idx = max([-1] + [int(d) for d in dirs if re.match(r'^[0-9]+$', d)])
    return max_idx + 1

def log(message, path, print_message=True):
    with open(path, 'a') as f:
        f.write(message)
        f.write("\n")
        if print_message:
            print(message)

### Training

def one_step(seq, model, opt, g_func, savepath, logpath, info, 
             gradient_clipping=False, i=0):
    """
    One gradient step.
    """
    # put sequence dim as 0th dim
    seq = seq.transpose(0, 1)
    bsize = seq.shape[1]

    mem0 = model.mem_init(bsize)
    main, contrastive = recurrent_apply_contrastive(model, seq, mem0)
    Loss_main = main.sum() / bsize
    Loss_contrastive = - g_func(contrastive).sum() / bsize
    Loss = Loss_main + Loss_contrastive

    Loss.backward()
    # rescale gradient if need be
    if gradient_clipping:
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

    # record gradients
    info['grads'].append(torch.cat(
        [p.grad.flatten() for p in model.parameters()]
    ))
    
    opt.step()
    opt.zero_grad()

    info['losses_main'].append(Loss_main.item())
    info['losses_contrastive'].append(Loss_contrastive.item())
    info['losses'].append(Loss.item())
    if i % 2 == 0:
        plt.plot(info['losses'], label='total')
        plt.plot(info['losses_main'], label='main')
        plt.plot(info['losses_contrastive'], label='contrastive')
        plt.legend(loc="upper right")

        plt.savefig(op.join(savepath, "train_loss.png"))
        plt.close()

    log(f"\tStep: {i}, Loss: {Loss.item()}", logpath)


def run_epoch(dl, model, opt, g_func, savepath, logpath, info, 
              gradient_clipping=False):
    """
    Train for one epoch on the dataset.

    info is a dict containing useful information for training, such as losses
    and gradients.
    """
    log("Beginning training loop.", logpath)
    for i, seq in enumerate(dl):
        print(f"loop {i}")
        one_step(seq, model, opt, g_func, savepath, logpath, info, 
                 gradient_clipping, i)

### First tesing utils

# get energies for normal and contrastive pairs
def constrast(model, dl):
    data = next(iter(dl))
    # complete

### Utilities

def save_model(model, savepath):
    torch.save(model.state_dict(), savepath)

def load_model(model, savepath):
    model.load_state_dict(torch.load(savepath))

def nparams(model):
    return sum(p.numel() for p in model.parameters())

def norm2(v):
    return ((v**2).sum())**.5

def cos_sim(v1, v2):
    # one-dimensional vectors
    return v1.dot(v2) / (norm2(v1) * norm2(v2))

### Run training

# if __name__ == "__main__":

# args = parser.parse_args()
EXPE_IDX = get_expe_idx()
if DEBUG:
    EXPE_IDX = "DEBUG"
if REDO_EXPE:
    EXPE_IDX -= 1

datapath = op.join("data", "two_spheres")
savepath = op.join("saves", args.task, str(EXPE_IDX))
logpath = op.join("saves", args.task, str(EXPE_IDX), "log.txt")

# create save directory if it doesn't exist
Path(savepath).mkdir(parents=True, exist_ok=True)

ds = ImageDs(path=datapath, seq_limit=100)
dl = DataLoader(ds, shuffle=True, batch_size=int(args.bsize))

# Define models and optimizer
model = CompleteModel_SlotDistance(
    K, F_MEM, HIDDEN_DIM, INPUT_DIMS, N_HEADS,  model_diff=True)
model2 = CompleteModel_SoftMatchingDistance(
    K, F_MEM, HIDDEN_DIM, INPUT_DIMS, N_HEADS)
model3 = CompleteModel_HardMatchingDistance(
    K, F_MEM, HIDDEN_DIM, INPUT_DIMS, N_HEADS)
opt = torch.optim.Adam(model.parameters(), lr=L_RATE)

# g_func = torch.nn.Identity()
g_func = lambda x: - BETA * F.softplus(1 - x)

data = next(iter(dl))
x1 = data[:, 0]
x2 = data[:, 1]
xc = data[:, 50]

# training metrics
info = {
    'losses': [],
    'losses_main': [],
    'losses_contrastive': [],
    'grads': []
}

def run():
    s = input(f"Please enter a short description for this run ({EXPE_IDX})")
    log(f"Experiment {EXPE_IDX}.", logpath)
    log(s, logpath)
    log("", logpath)

    today = datetime.datetime.today()
    log(f"beginning training, date:{today.day}/{today.month}/{today.year}, "
        f"{today.hour}:{today.minute}",
        logpath)
    log(f"task : {args.task}\n", logpath)


    for epoch in range(int(args.N)):
        log(f"\nbeginning epoch {epoch}\n", logpath)

        run_epoch(dl, model, opt, g_func, savepath, logpath, info)
        
        save_model(model, op.join(savepath, "model.pt"))
        log(
            f"model saved in directory {op.join(savepath, 'model.pt')}",
            logpath
        )

    log("\ntraining finished sucessfully", logpath)

def warm_start(model, seq, n):
    """
    Warmstarts the model with n iterations, on the given sequence.
    Returns the memory after N steps.
    """
    with torch.no_grad():
        mem = model.mem_init()

def plot_image(batch_idx, seq_idx, data=data):
    plt.imshow(data[batch_idx, seq_idx].transpose(0, 2).numpy()/2 + 0.5)
    plt.show()

def slot_sequence(model, data, t=20, model_diff=False):
    """
    Initializes a memory module, and visualize the slots over time as they
    incorporate info from the images.

    data is a batched sequence
    """
    # first show the sequence of images
    seq = data.transpose(0, 1)
    if not model_diff:
        fig, axs = plt.subplots(2, t)
    else:
        fig, axs = plt.subplots(3, t)

    seqr = seq[:, 0]

    # whitespace = torch.ones(3, 5, 100)
    # # represent only image in 0th position in minibatch
    # l = []
    # for i in range(t):
    #     l.append(seq[i])
    #     l.append(whitespace)
    # l.pop(-1)
    # imseq = torch.cat(l, dim=1)
    # imseq = imseq.transpose(0, 2)
    # imseq = imseq /2 + 0.5
    # axs[0].imshow(imseq)

    for i in range(t):
        axs[0, i].imshow(seqr[i].transpose(0, 2) / 2 + 0.5)

    one_batch_seq = seq[:, :1]
    mem = model.mem_init(1)
    memlist = [mem]
    with torch.no_grad():
        for i in range(t-1):
            d, mem = model(one_batch_seq[i], one_batch_seq[i+1], mem)
            axs[1, i].matshow(mem[0])

    if model_diff:
        with torch.no_grad():
            for i in range(t-1):
                d, mem = model(one_batch_seq[i], one_batch_seq[i+1], mem)

                axs[1, i].matshow(model.C_phi(one_batch_seq[i]) + mem[0])

    plt.show()

img0 = data[0, 0].unsqueeze(0)
img1 = data[0, 1].unsqueeze(0)
img2 = data[0, 2].unsqueeze(0)
img3 = data[0, 3].unsqueeze(0)
img4 = data[0, 4].unsqueeze(0)

seq = data.transpose(0, 1)
seq10 = seq[:10]
seq20 = seq[:20]
seq30 = seq[:30]
seq40 = seq[:40]
seq50 = seq[:50]
seq60 = seq[:60]
seq70 = seq[:70]
seq80 = seq[:80]
seq90 = seq[:90]
seq100 = seq[:100]
# load_model(model, "saves/two_spheres/1/model.pt")