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

from pathlib import Path
from torch.utils.data import DataLoader

# TODO use a model dict
from model.models import (CompleteModel_SlotDistance,
                          CompleteModel_SoftMatchingDistance,
                          CompleteModel_HardMatchingDistance,
                          recurrent_apply_contrastive)
from models.dataset import ImageDs

### Constants

N_EPOCHS = 10
BATCH_SIZE = 64
L_RATE = 1e-5
INPUT_DIMS = (50, 50, 3)
K = 4
F_MEM = 512
N_HEADS = 1

### Arguments for running training loops

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task',
                    dest='task',
                    default='two_spheres')
parser.add_argument('--N' '--n_epochs',
                    dest=N,
                    default=N_EPOCHS)
parser.add_argument('--B' '--batch_size',
                    dest=bsize,
                    default=BATCH_SIZE)

def log(message, path, print_message=True):
    with open(path, 'w') as f:
        f.write(message)
        f.write("\n")
        if print_message:
            print(message)

def run_epoch(dl, model, opt, g_func, logpath, losses):
    """
    Train for one epoch on the dataset.

    Takes in the list of losses and gradually appends the newly computed ones.
    """
    log("Beginning training loop.", logpath)
    for i, seq in enumerate(dl):
        opt.zero_grad()
        main, contrastive = recurrent_apply_contrastive(model, seq)
        Loss = main.sum(0) - g(contrastive.sum(0))

        Loss.backward()
        opt.step()

        losses.append(Loss.item())
        if i % 200 == 199:
            plt.plot(losses)
            plt.savefig(op.join(logpath, "train_loss.png"))
            plt.close()

        log(f"\tStep: {i}, Loss: {Loss.item()}", logpath)

def save_model(model, savepath):
    torch.save(model.state_dict(), savepath)

### Run training

if __name__ == "__main__":
    expe_idx = 0

    args = parser.parse_args()

    datapath = op.join("data", "two_sphere")
    savepath = op.join('saves', args.task, str(expe_idx))
    logpath = op.join('saves', args.task, str(expe_idx))

    # create save directory if it doesn't exist
    Path(savepath).mkdir(parents=True, exist_ok=True)

    today = datetime.datetime.today()
    log(f"beginning training, date:{today[2]}/{today[1]}/{today[0]}, "
        f"{today[3]}:{today[4]}",
        )
    log(f"task : {args.task}\n")

    ds = ImageDs(path=datapath)
    dl = DataLoader(ds, shuffle=True, batch_size=int(args.bsize))

    # Define models and optimizer
    model = CompleteModel_SlotDistance(
        BATCH_SIZE, K, F_MEM, INPUT_DIMS, N_HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=L_RATE)

    g_func = torch.nn.Identity() # TODO: change this

    # training metrics
    losses = []

    for epoch in range(int(args.N)):
        log(f"\nbeginning epoch {epoch}\n")

        run_epoch(dl, model, opt, g_func, logpath, losses)
        
        save_model(model, savepath)
        log(f"model saved in directory {savepath}")

    log("\ntraining finished sucessfully")