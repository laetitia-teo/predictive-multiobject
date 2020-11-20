"""
Partially adapted from C-SWMs.
"""

import torch
import torch.nn as nn
import numpy as np

import utils.utils as utils
import models.encoders as enc
import models.transitions as trs
import models.comparison as cmp
import models.util_models as utm

class Module(nn.Module):
    """
    This module defines the high-level model, containing the object extractor,
    object encoder, transition model, and comparison model.
    """
    def __init__(self, num_slots, slot_dim, hidden_dim, input_dims, num_heads, g_func, sigma=0.5,
                 extractor='cnn_small', relational=True, relation_type='transformer',
                 recurrent_transition=False, comparison='pairwise_l2', training='contrastive'):

        super().__init__()

        extractor = extractor.lower()
        relation_type = relation_type.lower()
        comparison = comparison.lower()
        training = training.lower()

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.input_dims = input_dims
        self.num_heads = num_heads
        self.g_func = g_func
        self.sigma = sigma

        self.extractor = extractor
        self.recurrent_transition = recurrent_transition
        self.training = training # used outside the model

        self.num_channels = input_dims[0]
        self.width_height = input_dims[1:]

        recurrent_encoder = False

        if self.extractor == 'cnn_small':
            self.obj_encoder = enc.EncoderCNNSmall(
                input_dim=self.num_channels,
                hidden_dim=hidden_dim // 16,
                num_slots=num_slots,
                slot_dim=slot_dim,
                width_height=self.width_height
            )
        elif self.extractor == 'cnn_medium':
            self.obj_encoder = enc.EncoderCNNMedium(
                input_dim=self.num_channels,
                hidden_dim=hidden_dim // 16,
                num_slots=num_slots,
                slot_dim=slot_dim,
                width_height=self.width_height
            )
        elif self.extractor == 'cnn_large':
            self.obj_encoder = enc.EncoderCNNLarge(
                input_dim=self.num_channels,
                hidden_dim=hidden_dim // 16,
                num_slots=num_slots,
                slot_dim=slot_dim,
                width_height=self.width_height
            )
        elif self.extractor == 'slot_attention':
            pass # TODO code and complete this

        # TODO code the recurrent transition models
        if not recurrent_transition and not relational:
            self.transition_model = trs.TransitionSimple(
                num_slots=num_slots,
                slot_dim=slot_dim,
                hidden_dim=hidden_dim
            )
        elif not recurrent_transition and relation_type == 'gnn':
            self.transition_model = trs.TransitionSimple_GNN(
                num_slots=num_slots,
                slot_dim=slot_dim,
                hidden_dim=hidden_dim
            )
        elif not recurrent_transition and relation_type == 'transformer':
            self.transition_model = trs.TransitionSimple_Transformer(
                num_slots=num_slots,
                slot_dim=slot_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=4
            )
        elif recurrent_transition and not relational:
            self.transition_model = trs.TransitionRecurrent(
                num_slots=num_slots,
                slot_dim=slot_dim,
                hidden_dim=hidden_dim,
            )
        elif recurrent_transition and relation_type == 'gnn':
            self.transition_model = trs.TransitionRecurrent_GNN(
                num_slots=num_slots,
                slot_dim=slot_dim,
                hidden_dim=hidden_dim,
            )
        elif recurrent_transition and relation_type == 'transformer':
            self.transition_model = trs.TransitionRecurrent_Transformer(
                num_slots=num_slots,
                slot_dim=slot_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=4
            )

        if comparison == 'pairwise_l2':
            self.comparison_model = cmp.PairwiseL2(
                scale=sigma
            )

        self.training = training

        if self.training == 'generative':
            # each encoder has its symmetrical decoder
            decoder_type = enc.encoder_decoder_map[type(self.obj_encoder)]
            self.obj_decoder = decoder_type(
                input_dim=self.num_channels,
                hidden_dim=hidden_dim // 16,
                num_slots=num_slots
            )

    def init_mem(self, batch_size):
        return self.transition_model.init_mem(batch_size)

    def compare_imgs(self, x, y):
        # L2 distance for now
        img_scale = 1.
        return ((x - y) / img_scale**2).pow(2).sum() / len(x)

    def contrastive_energy(self, x, x_next, slot_mem=None):
        # x, next_x :: [B. C, W, H]
        # contrasting is done by mixing up the batch
        random_inds = torch.randint(len(x), (len(x),))
        x_contrast = x[random_inds]

        slots = self.obj_encoder(x)
        slots_next = self.obj_encoder(x_next)
        slots_contrast = self.obj_encoder(x_contrast)

        slots_next_pred, next_mem = self.transition_model(slots, mem=slot_mem)

        positive = self.comparison_model(slots_next, slots_next_pred)
        negative = self.comparison_model(slots_contrast, slots) # TODO see for contrast here

        energy = positive - self.g_func(negative)
        energy = energy.sum() / len(x)

        positive = positive.sum() / len(x)
        negative = negative.sum() / len(x)

        # print(f"energy {energy}")
        # print(f"positive {positive}")
        # print(f"negative {negative}")
        # print(f"g_func negative {self.g_func(negative)}")
        # print()

        return energy, positive.item(), negative.item(), next_mem

    def generative_energy(self, x, x_next, slot_mem=None):
        slots = self.obj_encoder(x)
        slots_next_pred, next_mem = self.transition_model(slots, mem=slot_mem)
        x_next_decoded = self.obj_decoder(slots_next_pred)

        return self.compare_imgs(x_next, x_next_decoded), 0., 0., next_mem


    def compute_energy(self, x, x_next, slot_mem=None):

        if self.training == 'contrastive':
            energy, positive, negative, next_mem = self.contrastive_energy(x, x_next, slot_mem)
        elif self.training == 'generative':
            energy, positive, negative, next_mem = self.generative_energy(x, x_next, slot_mem)

        return energy, positive, negative, next_mem

    def compute_energy_sequence(self, seq):
        # seq :: [seq_len, batch_size, img_dims]
        seq_len = len(seq)
        batch_size = seq.shape[1]
        energies, positives, negatives = [], [], []

        mem = self.init_mem(batch_size)

        for i in range(seq_len - 1):
            x = seq[i]
            x_next = seq[i+1]
            energy, positive, negative, mem = self.compute_energy(x, x_next, slot_mem=mem)
            energies.append(energy)
            positives.append(positive)
            negatives.append(negative)

        energy_seq = sum(energies) / seq_len
        positive_seq = sum(positives) / seq_len
        negative_seq = sum(negatives) / seq_len

        # print(energy_seq)
        # print(positive_seq)
        # print(negative_seq)

        return energy_seq, positive_seq, negative_seq

if __name__ == "__main__":
    # test models
    seq_len = 5
    batch_size = 32
    num_slots = 4
    slot_dim = 2
    hidden_dim = 128
    num_heads = 2
    width, height = 30, 30
    num_channels = 6
    hinge = torch.tensor(1.)
    g_func = lambda x: 1. * torch.max(x, hinge)
    x = torch.rand(batch_size, num_channels, width, height)
    x_next = torch.rand(batch_size, num_channels, width, height)

    seq = torch.rand(seq_len, batch_size, num_channels, width, height)

    model = Module(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_dim=hidden_dim,
        input_dims=(num_channels, width, height),
        num_heads=num_heads,
        g_func=g_func,
        relational=False
    )
    E, P, N, _ = model.compute_energy(x, x_next)
    print(f"Energy, non-relational, non-recurrent: {E.item()}",
          f"\nPositive: {P}, Negative: {N}\n")
    res = model.compute_energy_sequence(seq)

    model = Module(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_dim=hidden_dim,
        input_dims=(num_channels, width, height),
        num_heads=num_heads,
        g_func=g_func,
        relational=True,
        relation_type='gnn'
    )
    E, _, _, _ = model.compute_energy(x, x_next)
    print(f"Energy, gnn relation type, non-recurrent: {E.item()}"
          f"\nPositive: {P}, Negative: {N}\n")
    res = model.compute_energy_sequence(seq)

    model = Module(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_dim=hidden_dim,
        input_dims=(num_channels, width, height),
        num_heads=num_heads,
        g_func=g_func,
        relational=True,
        relation_type='transformer'
    )
    E, _, _, _ = model.compute_energy(x, x_next)
    print(f"Energy, transformer relation type, non-recurrent: {E.item()}"
          f"\nPositive: {P}, Negative: {N}\n")
    res = model.compute_energy_sequence(seq)


