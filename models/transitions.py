import torch
import torch.nn as nn

import models.util_models as utm

class TransitionSimple(nn.Module):
    """
    Markovian transition model, with no information but the current state.
    No relations between nodes.
    """
    def __init__(self, num_slots, slot_dim, hidden_dim, **kwargs):
        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim

        self.node_model = utm.MLP_TwoLayers_Norm(
            input_size=slot_dim,
            hidden_size=hidden_dim,
            output_size=slot_dim
        )

    def init_mem(self, batch_size):
        return None

    def forward(self, x, *args, **kwargs):
        # x :: [B, K, F]
        return self.node_model(x), None

class TransitionSimple_GNN(nn.Module):
    """
    Markovian transition model, with no information but the current state.
    GNN-style relations.
    """
    def __init__(self, num_slots, slot_dim, hidden_dim, **kwargs):

        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim

        self.edge_model = utm.MLP_TwoLayers_Norm(
            input_size=2 * slot_dim,
            hidden_size=hidden_dim,
            output_size=hidden_dim
        )

        self.node_model = utm.MLP_TwoLayers_Norm(
            input_size=slot_dim + hidden_dim,
            hidden_size=hidden_dim,
            output_size=slot_dim
        )

    def init_mem(self, batch_size):
        return None

    def forward(self, x, *args, **kwargs):
        B, K, F = x.shape
        # create source and destination tensors
        # TODO: remove self-edges ?
        src = x.expand(K, B, K, F).permute(1, 2, 0, 3)
        dest = x.expand(K, B, K, F).transpose(0, 1)
        edges = torch.cat([src, dest], -1)
        edges = self.edge_model(edges)

        edge_agg = edges.sum(2) # TODO check dim of reduction

        slots = self.node_model(torch.cat([x, edge_agg], -1))
        return slots, None


class TransitionSimple_Transformer(nn.Module):
    """
    Markovian transition model, with no information but the current state.
    Transformer-style relations.
    """
    def __init__(self, num_slots, slot_dim, hidden_dim, num_heads,
                 num_layers, **kwargs):

        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # TODO: for now we simply embed the slots to higher dim, create
        # TODO: custom transformer layer

        # TODO: non-relational Transformer is not directly comparable
        # TODO: to non-relational GNN

        self.slot_encode = nn.Linear(slot_dim, hidden_dim)
        self.slot_decode = nn.Linear(hidden_dim, slot_dim)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=num_layers
        )

    def init_mem(self, batch_size):
        return None

    def forward(self, x, *args, **kwargs):
        # x :: [B, K, F], check this works with Transformer
        x = self.slot_encode(x)
        x = self.transformer(x)
        return self.slot_decode(x), None


# Models with internal memory

class TransitionRecurrent(nn.Module):
    """
    Non-Markovian transition model, additional memory information is passed
    across time-steps.
    The recurrence is implemented as a single hidden state per slot.
    Non-relational version.
    """
    def __init__(self, num_slots, slot_dim, hidden_dim, **kwargs):

        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim


    def init_mem(self, batch_size):
        return None

    def forward(self, x, mem):
        pass


class TransitionRecurrent_GNN(nn.Module):
    """
    Non-Markovian transition model, additional memory information is passed
    across time-steps.
    The recurrence is implemented as a single hidden state per slot.
    GNN-relational version.
    """
    def __init__(self, num_slots, slot_dim, hidden_dim, **kwargs):

        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim


    def init_mem(self, batch_size):
        return None

    def forward(self, x, mem):
        pass

class TransitionRecurrent_Transformer(nn.Module):
    """
        Non-Markovian transition model, additional memory information is passed
        across time-steps.
        The recurrence is implemented as a single hidden state per slot.
        Transformer version.
        """

    def __init__(self, num_slots, slot_dim, hidden_dim, **kwargs):
        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim


    def init_mem(self, batch_size):
        return None

    def forward(self, x, mem):
        pass