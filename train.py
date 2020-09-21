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
import os.path as op
import re
import csv
import json
import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader

import models.models as mod

# # TODO use a model dict
# from models.models import (CompleteModel_SlotDistance,
#                            CompleteModel_SoftMatchingDistance,
#                            CompleteModel_HardMatchingDistance,
#                            recurrent_apply_contrastive)
from models.dataset import ImageDs

### Constants

N_EPOCHS = 100
BATCH_SIZE = 16
L_RATE = 1e-3
INPUT_DIMS = (100, 100, 3)
K = 2
F_MEM = 2
HIDDEN_DIM = 512
N_HEADS = 1
# EXPE_IDX = 5
BETA = 1. # multiplicative factor for g function
CLIP_GRAD = 1e-3 # ?

MAX_SAMPLES = 64 # to control the overfitting regime

DEBUG = True # results go in DEBUG folder
REDO_EXPE = False # wether to redo previous experiment

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

def get_description():
    s = (f"\nF_MEM={F_MEM}\nHIDDEN_DIM={HIDDEN_DIM}\nL_RATE={L_RATE}\nN_EPOCH"
        f"S={N_EPOCHS}\nBATCH_SIZE={BATCH_SIZE}\n\n")
    return s

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
    main, contrastive = mod.recurrent_apply_contrastive(model, seq, mem0)
    Loss_main = main.sum() / bsize
    Loss_contrastive = - g_func(contrastive).sum() / bsize
    Loss = Loss_main + Loss_contrastive

    Loss.backward()
    # rescale gradient if need be
    if gradient_clipping:
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

    # record gradients
    info['grads'].append(get_grad(model))
    
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

def dump_info_json(info, savepath):
    s = json.dumps(info)
    with open("info.json", 'w') as jsonfile:
        jsonfile.write(s)

def dump_info_csv(info, savepath):
    # TODO: do in csv
    with open("info.csv", 'a') as csvfile:
        fieldnames = info.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # assumes all dict vales have the same length
        for i in range(len(info["losses"])):
            # wrier.writerow()
            pass 
            # TODO finish

def nparams(model):
    return sum(p.numel() for p in model.parameters())

def get_grad(model):
    # gets accumulated gradients in model parameters as a single vector
    pl = []
    for p in model.parameters():
        pl.append(p.grad.reshape(-1))
    return torch.cat(pl, 0)

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
elif REDO_EXPE:
    EXPE_IDX -= 1

datapath = op.join("data", "two_spheres")
savepath = op.join("saves", args.task, str(EXPE_IDX))
logpath = op.join("saves", args.task, str(EXPE_IDX), "log.txt")

ds = ImageDs(path=datapath, seq_limit=100, max_samples=MAX_SAMPLES)
dl = DataLoader(ds, shuffle=True, batch_size=int(args.bsize))
# datasets with one motionless ball
dstest = ImageDs(path=datapath, seq_limit=100, max_samples=MAX_SAMPLES, 
                 load_prefix="test")
dltest = DataLoader(dstest, shuffle=True, batch_size=int(args.bsize))

# Define models and optimizer
model = mod.CompleteModel_Debug(K, F_MEM, HIDDEN_DIM, (30, 30, 3), N_HEADS)
model1 = mod.CompleteModel_SlotDistance(
    K, F_MEM, HIDDEN_DIM, INPUT_DIMS, N_HEADS)
model2 = mod.CompleteModel_SoftMatchingDistance(
    K, F_MEM, HIDDEN_DIM, INPUT_DIMS, N_HEADS)
model3 = mod.CompleteModel_HardMatchingDistance(
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

def run(dataloader=None):
    # create save directory if it doesn't exist
    Path(savepath).mkdir(parents=True, exist_ok=True)

    if dataloader is None:
        dataloader = dl

    s = input(f"Please enter a short description for this run ({EXPE_IDX})")
    log(f"Experiment {EXPE_IDX}.", logpath)
    log(get_description(), logpath)
    log(s, logpath)
    log("", logpath)

    today = datetime.datetime.today()
    log(f"beginning training, date:{today.day}/{today.month}/{today.year}, "
        f"{today.hour}:{today.minute}",
        logpath)
    log(f"task : {args.task}\n", logpath)


    for epoch in range(int(args.N)):
        log(f"\nbeginning epoch {epoch}\n", logpath)

        run_epoch(dataloader, model, opt, g_func, savepath, logpath, info)
        
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

def plot_fmaps(model, data):
    # data [S, B, C, W, H]
    K = model.K
    W = model.W

    img = data[0, 0]
    img = img.permute(1, 2, 0)/2 + 0.5
    with torch.no_grad():
        fmap = model.C_phi.conv(data[0])[0].reshape(K, W//2, W//2)

    fig, axs = plt.subplots(1, K+1)
    axs[0].imshow(img)
    for i in range(K):
        axs[i+1].matshow(fmap[i])

    plt.show()

def plot_image_sequence(data):
    """
    Plots the sequence of images in a 9x9 grid.
    """
    im_size = 30
    vert_sep = np.ones((im_size, im_size//10, 3))
    hor_sep = np.ones((im_size//10, 9*im_size + 8*im_size//10, 3))

    data = data[:, 0]
    imgs = (data/2 + 0.5).transpose(1, 3).numpy()
    n_blank = max(81 - len(imgs), 0)
    img_blank = np.ones((im_size, im_size, 3))

    im_list = []

    for row in range(9):
        img_tot = np.zeros((im_size, 0, 3), dtype=np.float32)
        
        for col in range(9):
            if 9*row+col < len(imgs):
                img_tot = np.concatenate([img_tot, imgs[9*row+col]], axis=1)
            else:
                img_tot = np.concatenate([img_tot, img_blank], axis=1)

            if col < 9 - 1:
                img_tot = np.concatenate([img_tot, vert_sep], axis=1)

        if row < 9 - 1:
            img_tot = np.concatenate([img_tot, hor_sep], axis=0)

        im_list.append(img_tot)

    img_tot = np.concatenate(im_list, axis=0)

    plt.imshow(img_tot)
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False
    )
    plt.show()    

def plot_image_sequence_and_hidden_representations(model, data):
    """
    Given a complete model, a sequence of batched images, the function plots
    the sequence at batch index 0 and the scatterplot of representations.
    If the representations are not 2-dimensional, we do PCA/T-SNE/UMAP (TODO).
    The color of the points in the scatterplot give us the position in time
    of the representation. (The bluer they are, the more recent).

    The sequence is of length 81 (27*3), and is expected in [seq_len, batch,
    channel, width, height] format.
    """
    
    # compute image
    im_size = 30
    vert_sep = np.ones((im_size, im_size//10, 3))
    hor_sep = np.ones((im_size//5, 27*im_size + 26*im_size//10, 3))

    data = data[:, 0]
    imgs = (data/2 + 0.5).transpose(1, 3).numpy()
    n_blank = max(81 - len(imgs), 0)
    img_blank = np.ones((im_size, im_size, 3))

    im_list = []

    for row in range(3):
        img_tot = np.zeros((im_size, 0, 3), dtype=np.float32)
        
        for col in range(27):
            if 27*row+col < len(imgs):
                img_tot = np.concatenate([img_tot, imgs[27*row+col]], axis=1)
            else:
                img_tot = np.concatenate([img_tot, img_blank], axis=1)

            if col < 27 - 1:
                img_tot = np.concatenate([img_tot, vert_sep], axis=1)

        if row < 3 - 1:
            img_tot = np.concatenate([img_tot, hor_sep], axis=0)

        im_list.append(img_tot)

    img_tot = np.concatenate(im_list, axis=0)
    
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(img_tot)
    axs[0].tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False
    )

    # compute internal representations on the images
    with torch.no_grad():
        d_list, z_list, z_hat_list = model.forward_seq(data[:82].unsqueeze(1))

    # dimensionality reduction if the size of the features are bigger than 2
    if z_list[0].shape[-1] > 2:
        ... # reduce dimension with some algorithm
        raise Warning("dimension of the hidden states was reduced to 2 for"
                      "plotting")



    cm1 = plt.get_cmap("Oranges")
    colors1 = cm1(np.arange(81))
    xs1 = [z[0, 0, 0] for z in z_list]
    ys1 = [z[0, 0, 1] for z in z_list]

    axs[1].scatter(xs1, ys1, c=colors1)
    
    cm2 = plt.get_cmap("Reds")
    colors2 = cm2(np.arange(81))
    xs2 = [z[0, 1, 0] for z in z_list]
    ys2 = [z[0, 1, 1] for z in z_list]

    axs[1].scatter(xs2, ys2, c=colors2)

    cm3 = plt.get_cmap("Blues")
    colors3 = cm3(np.arange(81))
    xs3 = [z[0, 0, 0] for z in z_hat_list]
    ys3 = [z[0, 0, 1] for z in z_hat_list]

    axs[1].scatter(xs3, ys3, c=colors3)

    cm4 = plt.get_cmap("Greens")
    colors4 = cm4(np.arange(81))
    xs4 = [z[0, 1, 0] for z in z_hat_list]
    ys4 = [z[0, 1, 1] for z in z_hat_list]

    axs[1].scatter(xs4, ys4, c=colors4)

    plt.show()

def plot_grid_encoded(model, grid_idx=0):
    # load grid
    gridds = ImageDs(
        "data/two_spheres",
        seq_limit=100,
        max_samples=MAX_SAMPLES,
        load_prefix=f"grid{grid_idx}"
    )    
    data = gridds[0]
    with torch.no_grad():
        zs = model.C_phi(data)
    colors = np.arange(81)
    
    cm1 = plt.get_cmap("Blues")
    cm2 = plt.get_cmap("Oranges")

    z1s = zs[:, 0]
    z2s = zs[:, 1]

    plt.scatter(z1s[:, 0], z1s[:, 1], c=cm1(colors))
    plt.scatter(z2s[:, 0], z2s[:, 1], c=cm2(colors))

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