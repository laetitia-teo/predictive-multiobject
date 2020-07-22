"""
Defines the ParallelLSTM module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

### edge index utils

def get_crossgraph_ei(x, B, N1, N2):
    """
    Gets edge indices with source the vectors in batch1 and dest the vectors
    in batch2.
    B: number of batches, N1: number of objects in batch1, N2: number of
    objects in batch2.
    """
    BN1 = N1 * B
    BN2 = N2 * B
    
    src = torch.arange(BN1).expand(N2, BN1).transpose(0, 1).flatten()
    dest = torch.arange(BN2).expand(N1, BN2).view(N1, B, N2).transpose(0, 1)
    dest = dest.flatten() + BN1

    ei = torch.stack([src, dest], 0)
    return ei

#### sparse reduction ops

def scatter_sum(x, batch):
    nbatches = batch[-1] + 1
    nelems = len(batch)
    fx = x.shape[-1]
    i = torch.stack([batch, torch.arange(nelems)])
    
    st = torch.sparse.FloatTensor(
        i,
        x,
        torch.Size([nbatches, nelems] + list(x.shape[1:])),
    )
    return torch.sparse.sum(st, dim=1).values()

def scatter_mean(x, batch):
    nbatches = batch[-1] + 1
    nelems = len(batch)
    fx = x.shape[-1]
    i = torch.stack([batch, torch.arange(nelems)])
    
    st = torch.sparse.FloatTensor(
        i,
        x, 
        torch.Size([nbatches, nelems] + list(x.shape[1:])),
    )
    ost = torch.sparse.FloatTensor(
        i,
        torch.ones(nelems), 
        torch.Size([nbatches, nelems]),
    )
    xsum = torch.sparse.sum(st, dim=1).values()
    print(xsum.shape)
    nx = torch.sparse.sum(ost, dim=1).values().view([-1, 1])
    print(nx.shape)
    return xsum / nx

def scatter_softmax(x, batch):
    """
    Computes the softmax-reduction of elements of x as given by the batch index
    tensor.
    """
    nbatches = batch[-1] + 1
    nelems = len(batch)
    fx = x.shape[-1]
    i = torch.stack([batch, torch.arange(nelems)])
    
    # TODO: patch for numerical stability
    exp = x.exp()
    st = torch.sparse.FloatTensor(
        i,
        exp,
        torch.Size([nbatches, nelems] + list(x.shape[1:])),
    )
    expsum = torch.sparse.sum(st, dim=1).values()[batch]
    return exp / expsum

### Helpful submodules

class MLP(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        layers = []
        lin = layer_sizes.pop(0)

        for i, ls in enumerate(layer_sizes):
            layers.append(nn.Linear(lin, ls))
            if i < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
            ls = lin

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvNet(torch.nn.Module):
    def __init__(self, layers, mlp_layers=None, norm=False):
        super().__init__()

        layer_list = []

        for Fin, Fout, kernel_size in layers:
            layer_list.append(nn.Conv2d(Fin, Fout, kernel_size))
            # pooling ?
            layer_list.append(nn.MaxPool2d(2))
            layer_list.append(nn.ReLU())
        layer_list.pop(-1)

        if mlp_layers is not None:
            layer_list.append(nn.Flatten())
            for Fin, Fout in mlp_layers:
                layer_list.append(nn.Linear(Fin, Fout))
                layer_list.append(nn.ReLU())
            layer_list.pop(-1)

        if norm:
            # normalize (LayerNorm or BatchNorm ?)
            ...

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.conv(x)

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

        self.proj = nn.Linear(Fin, self.Ftot, bias=False)

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
        aw = torch.softmax(aw, dim=-1)

        out = (aw @ v)
        out = out.transpose(1, 2).reshape(B, N, self.Fv)

        return out

class AttentionLayerSparse(torch.nn.Module):
    """
    Sparse version of the above, for accepting batches with different
    numbers of objects.
    """
    def __init__(self, Fin, Fqk):
        super().__init__()
        self.Fin = Fin # in features
        self.Fqk = Fqk # features for dot product

        # for now values have the same dim as keys and queries
        self.Ftot = 2 * Fqk

        self.proj = nn.Linear(Fin, self.Ftot, bias=False)

    def forward(self, x, batch, ei):
        # remove dependence on batch ?

        src, dest = ei

        B = batch[-1] + 1

        scaling = float(self.Fqk) ** -0.5
        q, k = self.proj(x).chunk(2, -1)

        q = q * scaling

        qs, ks = q[src], k[dest]
        # dot product
        aw = qs.view(-1, 1, self.Fqk) @ ks.view(-1 self.Fqk, 1)
        aw = aw.squeeze()
        # softmax reduction
        aw = scatter_softmax(aw, src)

        # out = aw.view([-1, H, 1]) * vs
        # out = scatter_sum(out, src)
        # out = out.reshape([-1, self.Fv])

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
        self.linf = nn.Linear(2 * Fmem, Fmem)
        self.lini = nn.Linear(2 * Fmem, Fmem)
        self.linh = nn.Linear(2 * Fmem, Fmem)
        self.lino = nn.Linear(2 * Fmem, Fmem)

        self.proj = nn.Linear(2 * Fmem, 4 * Fmem)

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
    def __init__(self, Fin, Fqk, hard=True, tau=0.5):
        # note: no heads, maybe add ?
        # sparse implem for dealing with sparse edges ?
        self.Fin = Fin
        self.Fqk = Fqk
        self.Ftot = 2 * Fqk

        self.hard = hard
        # temperature for the gumbel-softmax
        self.tau = tau

        self.attention = AttentionLayerSparse()

    def forward(self, z, m):
        B, Nz, F = z.shape
        _, Nm, _ = m.shape
        zm = torch.cat([z, m], 1)

        batchz = torch.arange(B).expand(Nz, B).transpose(0, 1).flatten()
        batchm = torch.arange(B).expand(Nm, B).transpose(0, 1).flatten()
        batch = torch.cat([batchz, batchm], 0)

        ei = get_crossgraph_ei(B, Nz, Nm)
        
        # compute attention and sample edges
        aw = self.attention(zm, batch, ei)
        if self.hard:
            two_classes = torch.stack([aw, 1 - aw], -1)
            weight = F.gumbel_softmax(
                two_classes,
                hard=True,
                tau=self.tau)[:, 0]
        else:
            weight = aw

        src, dest = ei
        # distance weighed by the discrete edges/compatibility scores
        d = (z[src] - m[dest])**2 * weight

        return d

### Complete models

class BaseCompleteModel(torch.nn.Module):
    """
    Base class for a complete model, with an encoder, a slot-memory mechanism
    and a distance mechanism. Subclasses of this may define the models in the
    __init__ function directly.
    """    
    def __init__(self, C_phi, M_psi, Delta_xi, model_diff=False):
        super().__init__()

        self.C_phi = C_phi
        self.M_psi = M_psi
        self.Delta_xi = Delta_xi

    def next(self, x):
        return self.M_psi(self.C_phi(x))

    def forward(self, x1, x2):
        z1 = self.C_phi(x1)
        z2 = self.C_phi(x2)
        m = self.M_psi(z1)
        if not self.model_diff:
            d = self.Delta_xi(z2, m)
        else:
            alternative: model the difference
            d = self.Delta_xi(z2, z1 + m)

class CompleteModel_SlotDistance(BaseCompleteModel):
    """
    Slot-wise distance fn.
    """
    def __init__(self, B, K, Fmem, input_dims, nheads):

        self.H, self.W, self.C = input_dims
        # fix this to compute size of last vector
        C_phi = ConvNet(
            [
                (self.C, 32, 3),
                (32, 32, 3),
                (32, 32, 3),
                (32, 32, 3),    
            ],
            [
                (Fmem, Fmem),
                (Fmem, Fmem),
            ])
        M_psi = SelfAttentionLSTM_GNN(B, K, Fmem, nheads)
        Delta_xi = L2Dist()

        super().__init__(C_phi, M_psi, Delta_xi)

class CompleteModel_SoftMatchingDistance(BaseCompleteModel):
    """
    Simple CNN encoder;
    Parallel-LSTM with dot-product-attention communication between slots;
    Soft-slot-matching distance function.
    """
    def __init__(self, B, K, Fmem, input_dims, nheads):

        self.H, self.W, self.C = input_dims
        # fix this to compute size of last vector
        C_phi = ConvNet(
            [
                (self.C, 32, 3),
                (32, 32, 3),
                (32, 32, 3),
                (32, 32, 3),    
            ],
            [
                (Fmem, Fmem),
                (Fmem, Fmem),
            ])
        M_psi = SelfAttentionLSTM_GNN(B, K, Fmem, nheads)
        Delta_xi = MatchDistance(Fmem, Fmem, hard=False)

        super().__init__(C_phi, M_psi, Delta_xi)

class CompleteModel_HardMatchingDistance(BaseCompleteModel):
        """
    Simple CNN encoder;
    Parallel-LSTM with dot-product-attention communication between slots;
    Hard-slot-matching distance function.
    """
    def __init__(self, B, K, Fmem, input_dims, nheads):

        self.H, self.W, self.C = input_dims
        # fix this to compute size of last vector
        C_phi = ConvNet(
            [
                (self.C, 32, 3),
                (32, 32, 3),
                (32, 32, 3),
                (32, 32, 3),    
            ],
            [
                (Fmem, Fmem),
                (Fmem, Fmem),
            ])
        M_psi = SelfAttentionLSTM_GNN(B, K, Fmem, nheads)
        Delta_xi = MatchDistance(Fmem, Fmem, hard=True)

        super().__init__(C_phi, M_psi, Delta_xi)
