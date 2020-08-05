"""
The module that defines the testing procedures, such as the image recontruction
and distance regression tasks.

Maybe other tasks will come in the future.
"""

import torch

from model.models import modeldict
from model.util_models import ImageDecoder, BaselineVAE
from envs.multi_object_2d.multi_object_2d import envdict

N_EVAL = 1000 # number of evaluation samples

### Distance regression task

def distance_regression_test(env_name, model):
    for step in range(N_EVAL):
        
        env = envdict[env_name]
        

### Image reconstruction task

# TODO: code decoder model, have a pretrained vae to compare with