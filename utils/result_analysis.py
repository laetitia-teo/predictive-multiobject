"""
This utility file contains functions to analyse the data present in the save
folders.

The experiment data is organized as follows: the config files are in the configs
folder, and the models, as well as the train losses, are saved under the saves
folder. There is an option for adding a prefix, corresponding to data
"""
import re
import os
import os.path as op

import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import utils.utils as utl

from pprint import pprint
from torch.utils.data import DataLoader
from matplotlib.colors import TABLEAU_COLORS as COLORS

from models.models import Module
from utils.config_reader import ConfigReader
from utils.dataset import SequenceDataset

SAVE_DIR = 'saves'
CONFIG_DIR = 'configs'

class Analyser():
    """
    Result analysis object. Parses all config files, analyses which runs completed
    without error (by parsing the .err files), stores a dict of run parameters
    that also provides a pointer to the run directory. This dictionnary is used
    to  load train data and models, and to perform tests.
    """
    def __init__(self, prefix=''):
        self.prefix = prefix

        self.save_dir = op.join(SAVE_DIR, prefix)
        self.config_dir = op.join(CONFIG_DIR, prefix)

        self.config_dict = {}

        # list all directories
        c_dirs = os.listdir(self.config_dir)
        regex = r'config([0-9]+)'
        config_list = [
            re.search(regex, p)[1] for p in c_dirs if re.search(regex, p)
        ]

        s_dirs = os.listdir(self.save_dir)
        for c_idx in config_list:
            c_idx = int(c_idx)

            self.config_dict[c_idx] = {}

            # read all config params and store them in a dict
            self.config_dict[c_idx]['path'] = op.join(
                self.save_dir, f'config{c_idx}'
            )
            config = ConfigReader(op.join(self.config_dir, f'config{c_idx}'))
            for name, setting in config.settings.items():
                value = setting.get_value()
                self.config_dict[c_idx][name] = value

            # add the special OCCLUDER param
            self.config_dict[c_idx]['OCCLUDER'] = \
                ('occluder' in self.config_dict[c_idx]['EXPE'])

            # did the run complete without error ?
            self.config_dict[c_idx]['completed'] = 'yes'
            if prefix:
                # this means the results come from clusters and were computed
                # with slurm, so we can read the error logs
                err_log_path = op.join(self.save_dir, f'config{c_idx}_log.err')
                with open(err_log_path, 'r') as errf:
                    error_message = errf.readlines()
                    if error_message:
                        self.config_dict[c_idx]['completed'] = 'no'

            # check if model file and train data are present
            files = os.listdir(self.config_dict[c_idx]['path'])
            if 'model.pt' not in files:
                self.config_dict[c_idx]['completed'] = 'no'
            if 'train_data.hdf5' not in files:
                self.config_dict[c_idx]['completed'] = 'no'

            train_data = utl.load_dict_h5py(
                op.join(self.config_dict[c_idx]['path'], 'train_data.hdf5'))

            if len(train_data['energy']) != config.val('NUM_EPOCHS'):

                print(f'config {c_idx}')
                print(f'length of train data ({len(train_data["energy"])}) does'
                      f'not match number of epochs ({config.val("NUM_EPOCHS")})')

                self.config_dict[c_idx]['completed'] = 'partial'

    def plot_train_data_config(self, c_idx, show=True):
        try:
            path = op.join(self.config_dict[c_idx]['path'], 'train_data.hdf5')
            data_dict = utl.load_dict_h5py(path)
            plt.plot(data_dict['energy'], label='energy')
            plt.plot(data_dict['positive'], label='positive')
            plt.plot(data_dict['negative'], label='negative')
            plt.legend()

            pprint(self.config_dict[c_idx])

            if show:
                plt.show()

        except KeyError:
            print('Invalid config index')

    def print_failed_runs(self):
        print('Invalid runs:')
        for c_idx, val in self.config_dict.items():
            if val['completed'] == 'no':
                print(c_idx)

    def filter_config(self, c_idx, filter_dict):
        """
        Returns False if one of the key-value mappings in the filter_dict is not verified in the present
        config, True otherwise.

        Use this for filtering unwanted configs.
        """
        return True # TODO complete this

    def latent_space_grid(self, c_idx):
        """
        Loads the trained model in the config and plots the hidden states associated with
        the model.
        """
        num_channels = 6
        width = self.config_dict[c_idx]['WIDTH']
        height = self.config_dict[c_idx]['HEIGHT']
        model = Module(
            num_slots=self.config_dict[c_idx]['NUM_SLOTS'],
            slot_dim=self.config_dict[c_idx]['SLOT_DIM'],
            hidden_dim=self.config_dict[c_idx]['HIDDEN_DIM'],
            input_dims=(num_channels, width, height),
            num_heads=self.config_dict[c_idx]['NUM_HEADS'],
            g_func=self.config_dict[c_idx]['g_func'],
            relational=self.config_dict[c_idx]['RELATIONAL'],
            relation_type=self.config_dict[c_idx]['RELATION_TYPE'],
            recurrent_transition=self.config_dict[c_idx]['RECURRENT_TRANSITION'],
            training=self.config_dict[c_idx]['TRAINING'],
        )

        if 'two_sphere' in self.config_dict[c_idx]['EXPE']:
            grid_ds = SequenceDataset(op.join(SAVE_DIR, "two_sphere_grid.hdf5"))
            grid_dl = DataLoader(
                grid_ds,
                shuffle=True,
                batch_size=81,
                num_workers=4
            )
            num_obj = 2
        elif 'three_body' in self.config_dict[c_idx]['EXPE']:
            pass # TODO complete this

        fig, axs = plt.subplot(num_obj)

        data = next(iter(grid_dl))

        for obj_idx in range(num_obj):
            slots = model.obj_encoder(data)
            axs[obj_idx].plot([])
            # TODO complete this

    def plot_train_data_separate_by_param(self, param_name, filter_dict={}, show=True):
        """
        This function takes the name of a parameter in the config, and plots each train energy corresponding
        to one value of the parameter in one particular color.
        """
        colorlist = list(COLORS.keys())
        param_values = []
        legend_handles = []
        for c_idx, config in self.config_dict.items():

            if not self.filter_config(c_idx, filter_dict):
                continue

            if config[param_name] not in param_values:
                param_values.append(config[param_name])

            color = colorlist[param_values.index(config[param_name])]
            print(f'color 2 {color}')
            path = op.join(self.config_dict[c_idx]['path'], 'train_data.hdf5')
            data_dict = utl.load_dict_h5py(path)
            plt.plot(data_dict['energy'], color=color)

        # legend
        for color, value in zip(colorlist, param_values):
            print(f"COLOR {color}")
            line = mlines.Line2D([], [], color=color, label=f'{param_name}: {value}')
            legend_handles.append(line)

        plt.legend(legend_handles, param_values)

        if show:
            plt.show()