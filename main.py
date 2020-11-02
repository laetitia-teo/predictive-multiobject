# import models.models_ as mod
#
# from train import run, dl
#
# def main():
#     run(dl) # add hparams
#
# if __name__ == "__main__":
#     main()

import os, sys
import os.path as op
import datetime
import csv
import torch

import matplotlib.pyplot as plt

import utils.utils as utl

from torch.utils.data import DataLoader

from utils.dataset import SequenceDataset
from models.models import Module
from utils.config_reader import ConfigReader
# from utils.utils import save_dict_h5py

num_args = len(sys.argv) - 1

# if num_args != 1:
#     print('run.py accepts a single argument specifying the config file.')
#     exit(1)

# Read the config file
# config = ConfigReader(sys.argv[1])
config_id = 0

config = ConfigReader(f"configs/config{config_id}.txt")

RELATIONAL = config.val("RELATIONAL")
RELATION_TYPE = config.val("RELATION_TYPE")
RECURRENT_TRANSITION = config.val("RECURRENT_TRANSITION")
TRAINING = config.val("TRAINING")
G_FUNC = config.val("G_FUNC")
HINGE = torch.tensor(config.val("HINGE"))

NUM_SLOTS = config.val("NUM_SLOTS")
SLOT_DIM = config.val("SLOT_DIM")
HIDDEN_DIM = config.val("HIDDEN_DIM")
NUM_HEADS = config.val("NUM_HEADS")

EXPE = config.val("EXPE")
NUM_EPOCHS = config.val("NUM_EPOCHS")
LEARNING_RATE = config.val("LEARNING_RATE")
BATCH_SIZE = config.val("BATCH_SIZE")
NUM_WORKERS = 4
WIDTH = config.val("WIDTH")
HEIGHT = WIDTH
NUM_CHANNELS = 6
SAVE_PATH = config.val("SAVE_PATH")

if G_FUNC == 'hinge':
    g_func = lambda x: 1. * torch.min(x, HINGE)
else:
    raise ValueError("unknowns g_func type")

model = Module(
    num_slots=NUM_SLOTS,
    slot_dim=SLOT_DIM,
    hidden_dim=HIDDEN_DIM,
    input_dims=(NUM_CHANNELS, WIDTH, HEIGHT),
    num_heads=NUM_HEADS,
    g_func=g_func,
    relational=RELATIONAL,
    relation_type=RELATION_TYPE,
    recurrent_transition=RECURRENT_TRANSITION,
    training=TRAINING
)

opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

dataset = SequenceDataset(op.join("data", f"{EXPE}.hdf5"))
dataloader = DataLoader(
    dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# training loop

train_data = {
    'energy': [],
    'positive': [],
    'negative': [],
    'grads': []
}

t = datetime.datetime.utcnow()
experiment_name = (f"config{config_id}_{t.year}_{t.month}_{t.day}_{t.hour}_"
                   f"{t.minute}_{t.microsecond}")
# make folder with experiment name

train_data_save_path = op.join(
    SAVE_PATH,
    experiment_name,
    "train_data.hdf5"
)

model_save_path = op.join(
    SAVE_PATH,
    experiment_name,
    "model.pt"
)

try:
    for epoch in range(NUM_EPOCHS):

        print(f"Epoch {epoch}")

        for batch, data in enumerate(dataloader):

            opt.zero_grad()
            energy, positive, negative = model.compute_energy_sequence(data)
            energy.backward()
            opt.step()

            train_data['energy'].append(energy.item())
            train_data['positive'].append(positive)
            train_data['negative'].append(negative)
            # train_data['grads'].append(0.)

            print(f"\tBatch {batch}, loss {energy.item()}, positive {positive},",
                  f"negative {negative}")

    utl.save_plot_dict(
        train_data,
        op.join(SAVE_PATH, experiment_name, "train_plot.png")
    )
    utl.save_model(model, model_save_path)
    utl.save_dict_h5py(train_data, train_data_save_path)

except Exception:
    # an error occured
    utl.save_plot_dict(
        train_data,
        op.join(SAVE_PATH, experiment_name, "train_plot.png")
    )
    utl.save_model(model, model_save_path)
    utl.save_dict_h5py(train_data, train_data_save_path)
    raise Exception