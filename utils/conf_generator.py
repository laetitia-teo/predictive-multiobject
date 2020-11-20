"""
This file is provided with enums with all values to explore for the different parameters
of the experiment. It generates config files with all possible combinations of the
enumerated parameters.
"""

import sys
import re
import random
import os
import os.path as op

sys.path.append('..')

# number of seeds
num_seeds = 2
# SEEDS = [random.randint(0, 10000) for _ in range(num_seeds)]
SEEDS = list(range(num_seeds))

# fixed arguments

NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 64

# argument enums
RELATIONAL = ['False']
RELATION_TYPE = ['GNN']
RECURRENT_TRANSITION = ['False']
TRAINING = ['contrastive']
G_FUNC = ['hinge']
HINGE = ['1.']

EXPE = [
    'two_sphere_simple',
    'two_sphere_indep',
    'two_sphere_indep_partial',
    'three_body_physics'
]
OCCLUDER = [False, True]

# generate configs

config_folder = '../configs'

# get larger config number

# paths = os.listdir()
paths = os.listdir(config_folder)
idx_list = [re.search(r'config([0-9]+)', p)[1] for p in paths
            if re.search(r'config([0-9]+)', p)]
if idx_list:
    max_expe_idx = max(idx_list)

idx = 0

for seed_value in SEEDS:
    for relational_value in RELATIONAL:
        for relation_type_value in RELATION_TYPE:
            for expe_value in EXPE:
                for occluder_value in OCCLUDER:

                    print(idx)
                    config_name = op.join(config_folder, f'config{idx}')

                    if occluder_value:
                        expe_value += '_occluder'

                    text =  (
                        '### Model type ###\n'
                        '\n'
                        f'RELATIONAL = {relational_value}\n'
                        f'RELATION_TYPE = {relation_type_value}\n'
                        'RECURRENT_TRANSITION = False\n'
                        'TRAINING = contrastive\n'
                        'G_FUNC = hinge\n'
                        'HINGE = 1.\n'
                        '\n'
                        '### Model params ###\n'
                        '\n'
                        'NUM_SLOTS = 4\n'
                        'SLOT_DIM = 2\n'
                        'HIDDEN_DIM = 128\n'
                        'NUM_HEADS = 2\n'
                        '\n'
                        '### Experiment params ###\n'
                        '\n'
                        f'EXPE = {expe_value}\n'
                        f'SEED = {seed_value}\n'
                        '\n'
                        f'NUM_EPOCHS = {NUM_EPOCHS}\n'
                        f'BATCH_SIZE = {BATCH_SIZE}\n'
                        f'LEARNING_RATE = {LEARNING_RATE}\n'
                        'WIDTH = 30 # maybe not useful\n'
                        'HEIGHT = 30 # maybe not useful\n'
                        '\n'
                        'SAVE_PATH = saves\n'
                    )

                    print(config_name)
                    with open(config_name, 'w') as f:
                        f.write(text)

                    idx += 1