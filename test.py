"""
The module that defines the testing procedures.

Baseline ideas:

    - Same model with only one slot (have more "combinatoriality" in the train
      datasets, the difference with the single-slotted model will be more 
      pronounced)
    - Generative model (compare in terms of sample efficiency)
    - SimCore (this one is generative, single-slot)

The testing tasks are the following:

1) Image reconstruction.

This set of tasks tests the static encoding properties of our model.

For this first task, we train a decoder on the hidden representations of
interest. We compare the empirical mean of the reconstruction score over
a set of test images with the one given by a standard vae baseline.

We examine the disentanglement properties of the hidden representations. For
this, there are several dimensions of interest: 

    * Are the different slots representing different objects ?

      To test this, we may feed in empty images to produce a set of information-
      less representations Ei, and replace some of the slots produced by an
      encoding of a state by some of the Ei to see how the reconstruction is
      afected by this. Another option could be to duplicate some of the slots
      and visually inspect the reconstructions.

    * Are the features in each slot disentangled ?

      For testing this, classical traversal visualisations could do. If the
      above point is also true, we can inspect each feature of each object
      independently.

A note here: if we want to have disentangled features for each object we may
want to reduce the number of features of the memory slots.

Another note on this task: one of the advantages of using a discriminative,
contrastive loss, is to be able to take into account objects that are small in
the visual field. A classical generative approach could miss it since not
reconstructing it does not penalize the model a lot. Would a decoder trained on
our contrastively-trained latent representations also have this property ?
We could explore this by looking at the qualitative reconstruction of small
objects.

NB: maybe have a decoder that can take a variable number of objects as input. 
This can allow us to extrapolate to more objects in decoder. For instance, 
spatial broadcast decoder where each pixel in the original feature map attends
to each slot with a self-attention mechanism.

2) Reconstruction of the dynamics

Once we have trained an image decoder from the latent vectors, we can use it to
inspect the learned dynamics of the model by rolling out the internal 
memory/dynamics model and decoding along the way. This can allow us to see the
evolution of the internal state as interpreted by the memory. Compare with
baselines.

3) Distance regression

This is an example of a relational task to be done on the hidden 
representations, maybe some other tasks could be considered, such as:

    * Question-Answering (relational, counting, existence) (for more complex
      settings)
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