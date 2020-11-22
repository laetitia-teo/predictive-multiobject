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
import utils.utils as utl

from utils.config_reader import ConfigReader

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

            self.config_dict[c_idx] = {}

            # read all config params
            self.config_dict[c_idx]['path'] = op.join(
                self.save_dir, f'config{c_idx}'
            )
            config = ConfigReader(op.join(self.config_dir, f'config{c_idx}'))

            for name, setting in config.settings.items():
                value = setting.get_value()
                config[name] = value

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

