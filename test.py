"""
Testing module.
"""
import os.path as op

from torch.utils.data import DataLoader

from utils.result_analysis import Analyser
from utils.dataset import SequenceDataset

analyser = Analyser(prefix='plafrim')
analyser.plot_train_data_separate_by_param('EXPE')
None

dataset = SequenceDataset(op.join("data", "two_sphere_grid.hdf5"))
dataloader = DataLoader(
    dataset,
    shuffle=True,
    batch_size=81,
    num_workers=4
)
None