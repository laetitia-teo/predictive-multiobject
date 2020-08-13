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

import datetime
import os.path as op
import argparse

import matplotlib.pyplot as plt

import torch
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

N_EPOCHS = 10
BATCH_SIZE = 64
L_RATE = 1e-4
INPUT_DIMS = (100, 100, 3)
K = 4
F_MEM = 512
N_HEADS = 1
EXPE_IDX = 1
BETA = 1. # multiplicative factor for g function

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

### Training

def log(message, path, print_message=True):
    with open(path, 'w') as f:
        f.write(message)
        f.write("\n")
        if print_message:
            print(message)

def run_epoch(dl, model, opt, g_func, savepath, logpath, losses):
    """
    Train for one epoch on the dataset.

    Takes in the list of losses and gradually appends the newly computed ones.
    """
    log("Beginning training loop.", logpath)
    for i, seq in enumerate(dl):
        print(f"loop {i}")

        # put sequence dim as 0th dim
        seq = seq.transpose(0, 1)
        bsize = seq.shape[1]

        opt.zero_grad()
        mem0 = model.mem_init(bsize)
        main, contrastive = recurrent_apply_contrastive(model, seq, mem0)
        Loss = main.sum() - g_func(contrastive.sum())
        print(Loss.shape)
        print(Loss)

        Loss.backward()
        opt.step()

        losses.append(Loss.item())
        if i % 2 == 0:
            plt.plot(losses)
            plt.savefig(op.join(savepath, "train_loss.png"))
            plt.close()

        log(f"\tStep: {i}, Loss: {Loss.item()}", logpath)

### Utilities

def save_model(model, savepath):
    torch.save(model.state_dict(), savepath)

def load_model(model, savepath):
    model.load_state_dict(torch.load(savepath))

### Run training

# if __name__ == "__main__":

args = parser.parse_args()

datapath = op.join("data", "two_sphere")
savepath = op.join("saves", args.task, str(EXPE_IDX))
logpath = op.join("saves", args.task, str(EXPE_IDX), "log.txt")

# create save directory if it doesn't exist
Path(savepath).mkdir(parents=True, exist_ok=True)

today = datetime.datetime.today()
log(f"beginning training, date:{today.day}/{today.month}/{today.year}, "
    f"{today.hour}:{today.minute}",
    logpath)
log(f"task : {args.task}\n", logpath)

ds = ImageDs(path=datapath)
dl = DataLoader(ds, shuffle=True, batch_size=int(args.bsize))

# Define models and optimizer
model = CompleteModel_SlotDistance(
    K, F_MEM, INPUT_DIMS, N_HEADS)
model2 = CompleteModel_SoftMatchingDistance(
    K, F_MEM, INPUT_DIMS, N_HEADS)
model3 = CompleteModel_HardMatchingDistance(
    K, F_MEM, INPUT_DIMS, N_HEADS)
opt = torch.optim.Adam(model.parameters(), lr=L_RATE)

# g_func = torch.nn.Identity()
g_func = lambda x: BETA * F.softplus(1 - x)

data = next(iter(dl))
x1 = data[:, 0]
x2 = data[:, 1]

# training metrics
losses = []

def run():
    for epoch in range(int(args.N)):
        log(f"\nbeginning epoch {epoch}\n", logpath)

        run_epoch(dl, model, opt, g_func, savepath, logpath, losses)
        
        save_model(model, op.join(savepath, "model.pt"))
        log(f"model saved in directory {savepath}", logpath)

    log("\ntraining finished sucessfully", logpath)

def plot_image(batch_idx, seq_idx, data=data):
    plt.imshow(data[batch_idx, seq_idx].transpose(0, 2).numpy()/2 + 0.5)
    plt.show()

img0 = data[0, 0].unsqueeze(0)
img4 = data[0, 4].unsqueeze(0)

load_model(model, "saves/two_spheres/1/model.pt")