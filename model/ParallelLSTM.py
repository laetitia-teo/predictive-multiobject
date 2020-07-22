"""
Defines the ParallelLSTM module.
"""

import torch

from torch.nn import Linear, Sequential, ReLU

### Helpful submodules

class MLP(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        layers = []
        lin = layer_sizes.pop(0)

        for i, ls in enumerate(layer_sizes):
            layers.append(Linear(lin, ls))
            if i < len(layer_sizes) - 1:
                layers.append(ReLU())
            ls = lin

        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SelfAttentionLayer(torch.nn.Module):
    """
    Multi-head Self-Attention layer.

    Inputs:
        - x: the input data for a minibatch, concatenated in the 0-th dim.
        - batch: an index tensor giving the indices of the elements of x in
            minibatch.
    """
    def __init__(self, Fin, Fqk, Fv, nheads):
        super().__init__()
        self.Fin = Fin # in features
        self.Fqk = Fqk # features for dot product
        self.Fv = Fv # features for values
        self.nheads = nheads

        assert Fqk // nheads == Fqk / nheads, "self-attention features must "\
            " be divisible by number of heads"
        assert Fv // nheads == Fv / nheads, "value features must "\
            " be divisible by number of heads"

        # for now values have the same dim as keys and queries
        self.Ftot = (2*Fqk + Fv)

        self.proj = Linear(Fin, self.Ftot, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        H = self.nheads
        Fh = self.Fqk // H
        Fhv = self.Fv // H

        scaling = float(Fh) ** -0.5
        q, k, v = self.proj(x).split([self.Fqk, self.Fqk, self.Fv], dim=-1)

        q = q * scaling
        q = q.reshape(B, N, H, Fh).transpose(1, 2)
        k = k.reshape(B, N, H, Fh).transpose(1, 2)
        v = v.reshape(B, N, H, Fhv).transpose(1, 2)

        aw = q @ (k.transpose(2, 3))
        aw = torch.softmax(aw, dim=-2)

        out = (aw @ v)
        out = out.transpose(1, 2).reshape(B, N, self.Fv)

        return out

class TransformerBlock(torch.nn.Module):
    """
    Implements a full Transformer block, with skip connexions, layernorm
    and an mlp.

    Arguments:
        - d: dimension of a head
        - h: number of heads
    """
    def __init__(self, d, h):
        super().__init__()

        self.d = d
        self.h = h

        self.norm1 = torch.nn.LayerNorm([d])
        self.norm2 = torch.nn.LayerNorm([d])

        self.mhsa = SelfAttentionLayer(d, d, d, h)
        # TODO: check papers for hparams
        self.mlp = MLP([d, d, d])

    def forward(self, x):

        y = self.mhsa(x)
        y = self.norm1(x + y)

        z = self.mlp(y)
        z = self.norm2(y + z)

        return z

### Slot-Memory architectures

class SelfAttentionLSTM_GNN(torch.nn.Module):
    """
    A GNN where the edge model + edge aggreg is a self-attention layer.
    There are K hidden states and cells, each corresponding to a particular
    memory slot. The LSTN parameters are shared between all slots.

    Dense implem (should we call this a GNN ?)

    We have two choices for what vectors we use for the self-attention update:
    hidden vectors of cells. We'll use cells here, but that may not be the best
    choice.
    """
    def __init__(self, B, K, Fmem, nheads):
        super().__init__()

        self.B = B
        self.K = K
        self.Fmem = Fmem
        self.nheads = nheads

        # maybe replace with something else
        self.self_attention = TransformerBlock(
            Fmem,
            nheads,
        )
        self.linf = Linear(2 * Fmem, Fmem)
        self.lini = Linear(2 * Fmem, Fmem)
        self.linh = Linear(2 * Fmem, Fmem)
        self.lino = Linear(2 * Fmem, Fmem)

        self.proj = Linear(2 * Fmem, 4 * Fmem)

        C, H = self._mem_init()
        self.register_buffer('C', C)
        self.register_buffer('H', H)

    def _mem_init(self):
        """
        Some form of initialization where the vectors are unique.
        """
        C = torch.cat([
            torch.eye(self.K).expand([self.B, self.K, self.K]),
            torch.zeros([self.B, self.K, self.Fmem - self.K])
        ], -1)
        H = torch.cat([
            torch.eye(self.K).expand([self.B, self.K, self.K]),
            torch.zeros([self.B, self.K, self.Fmem - self.K])
        ], -1)
        return C, H

    def forward(self, x):

        C_cat = torch.cat([self.C, x], 1)
        # input to the slot-LSTM
        X = self.self_attention(C_cat)[:, :self.K]

        HX = torch.cat([self.H, X], -1)
        f, i, o, Ctilde = self.proj(HX).chunk(4, -1)

        # note: no tanh in content update and output
        C = self.C * torch.sigmoid(f) + Ctilde * torch.sigmoid(i)
        H = C * torch.sigmoid(o)

        self.register_buffer('C', C)
        self.register_buffer('H', H)

        return H

### Slot-distance functions

class L2Dist(torch.nn.Module):
    """
    Simple slot-wise L2 distance.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z, m):
        return (z - m)**2

class NegativeCosSim(torch.nn.Module):
    """
    Slot-wise negative cosine similarity.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z, m):
        # z, m :: [B, N, F]
        # TODO check this
        B, N, F = z.shape
        normz = z**2.sum(-1).unsqueeze(-1)
        normm = m**2.sum(-1).unsqueeze(-1)

        dot = z.view([B, N, 1, F]) @ m.view([B, N, F, 1])
        return dot.squeeze(-1) / (normz * normm)

class MatchDistance(torch.nn.Module):
    """
    This module defines a matching procedure between the two given slot-based
    elements: First a compatibility score is computed between all vectors and
    constrained to be positive and sum to 1 on all destination vectors.
    These compatibility scores are then used as the parameters of a Bernoulli
    over each pair of vectors, which is then sampled to match vectors. The
    constraint implements a form of competition between vectors of the second
    input.
    The distance computation is then made according to those created matchings.
    """
    def __init__(self, Fin, Fqk):
        # note: no heads, maybe add ?
        # sparse implem for dealing with sparse edges ?
        self.Fin = Fin
        self.Fqk = Fqk
        self.Ftot = 2 * Fqk

        self.proj = Linear(Fin, self.Ftot, bias=False)

    def forward(self, z, m):
        B, Nz, F = z.shape
        _, Nm, _ = m.shape
        # compute q and k separately (?)
        q, k = self.proj()