import argparse
import utils
import datetime
import os.path as op
import pickle

import torch
import torch.nn.functional as F

import numpy as np
import logging

import models.modules

# from torch.utils import data
from torch.utils.data import DataLoader

from utils.dataset import TransitionDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')

parser.add_argument('--encoder', type=str, default='small',
                    help='Object extrator CNN size (e.g., `small`).')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')

parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=2,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Dimensionality of action space.')
parser.add_argument('--num-objects', type=int, default=5,
                    help='Number of object slots in model.')
parser.add_argument('--ignore-action', action='store_true', default=False,
                    help='Ignore action in GNN transition model.')
parser.add_argument('--copy-action', action='store_true', default=False,
                    help='Apply same action to all object slots.')

parser.add_argument('--decoder', action='store_true', default=False,
                    help='Train model using decoder and pixel-based loss.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=20,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--dataset', type=str,
                    default='data/balls_train.h5',
                    help='Path to replay buffer.')
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

now = datetime.datetime.now()
timestamp = now.isoformat()

# if args.name == 'none':
#     exp_name = timestamp
# else:
#     exp_name = args.name

# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

# exp_counter = 0
# save_folder = '{}/{}/'.format(args.save_folder, exp_name)

# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# meta_file = os.path.join(save_folder, 'metadata.pkl')
# model_file = os.path.join(save_folder, 'model.pt')
# log_file = os.path.join(save_folder, 'log.txt')

# logging.basicConfig(level=logging.INFO, format='%(message)s')
# logger = logging.getLogger()
# logger.addHandler(logging.FileHandler(log_file, 'a'))
# print = logger.info

# pickle.dump({'args': args}, open(meta_file, "wb"))

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = TransitionDataset("envs/multi_object_2d/test.hdf5")
dl = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=4)

sample = next(iter(dl))
input_shape = sample[0].size()[1:]

model = models.modules.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    num_objects=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action,
    encoder=args.encoder).to(device)

# model.apply(utils.weights_init)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate)

# Train loop
print('Starting model training...')
step = 0
best_loss = 1e9
losses = []

# for epoch in range(1, args.epochs + 1):
#     model.train()
#     train_loss = 0
#
#     for batch_idx, data_batch in enumerate(dl):
#         data_batch = [tensor.to(device) for tensor in data_batch]
#         optimizer.zero_grad()
#
#         loss = model.contrastive_loss(*data_batch)
#
#         loss.backward()
#         train_loss += loss.item()
#         losses.append(loss.item())
#         optimizer.step()
#
#         if batch_idx % args.log_interval == 0:
#             print(
#                 'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(data_batch[0]),
#                     len(dl.dataset),
#                     100. * batch_idx / len(dl),
#                     loss.item() / len(data_batch[0])))
#
#         step += 1
#
#     avg_loss = train_loss / len(dl.dataset)
#     print('====> Epoch: {} Average loss: {:.6f}'.format(
#         epoch, avg_loss))
#
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         torch.save(model.state_dict(), op.join(
#             "saves", "cswm_debug", "model.pt"))
#
