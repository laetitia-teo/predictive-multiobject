"""
File for generating the datasets used in the simple setups.
"""

import argparse
import numpy as np

from utils.utils import save_list_dict_h5py

from ..envs.multi_object_2d.multi_object_2d import generate_two_sphere_dataset
from ..envs.three_body_physics.physics_sim import \
    generate_3_body_problem_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str,
                    default='data',
                    help='File name / path.')
parser.add_argument('--num-episodes', type=int, default=1000,
                    help='Number of episodes to generate.')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')
parser.add_argument('--eval', action='store_true', default=False,
                    help='Create evaluation set.')
parser.add_argument('--task', type=str,
                    default='two-sphere-simple',
                    help='What dataset to generate.')
parser.add_argument('--seq-len', type=int, default=20,
                    help='Length of the generated sequences.')


args = parser.parse_args()

np.random.seed(args.seed)

if args.task == "two-sphere-simple":
    generate_two_sphere_dataset(
        dest=args.fname,
        train_set_size=args.num_episodes,
        mode="simple",
        img_size=[30, 30],
        dt=0.3,
        seed=args.seed
    )
elif args.task == "two-sphere-indep":
    generate_two_sphere_dataset(
        dest=args.fname,
        train_set_size=args.num_episodes,
        mode="indep",
        img_size=[30, 30],
        dt=0.3,
        seed=args.seed
    )
elif args.task == "two-sphere-indep-partial":
    generate_two_sphere_dataset(
        dest=args.fname,
        train_set_size=args.num_episodes,
        mode="indep_partial",
        img_size=[30, 30],
        dt=0.3,
        seed=args.seed
    )
elif args.task == "tree-body-physics":
    generate_3_body_problem_dataset(
        dest=args.fname + '.npz',
        train_set_size=args.num_episodes,
        valid_set_size=2,
        test_set_size=2,
        seq_len=args.seq_len,
        img_size=[50, 50],
        dt=2.0,
        vx0_max=0.5,
        vy0_max=0.5,
        color=True,
        seed=args.seed
    )