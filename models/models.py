"""
Defines the models used for the experiments.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import util_models as utm
except ModuleNotFoundError:
    import models.util_models as utm


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

### Encoders

class PureCNNEncoder(nn.Module):
    """
    Simplest encoder with CNN + chunking on the feature dim.
    """
    def __init__(self, in_ch, inter_ch, out_ch, in_size, K, stop=0):
        super().__init__()

        self.K = K

        self.conv = utm.MaxpoolEncoder(
            in_ch, inter_ch, out_ch * K, in_size, stop=stop)
    
    def forward(self, x):
        z = self.conv(x)
        # format by slots
        zs = torch.stack(z.chunk(self.K, -1), 1)
        return zs

class PureCNNEncoder_nopool(nn.Module):
    """
    Simplest encoder with CNN + chunking on the feature dim.
    """
    def __init__(self, in_ch, inter_ch, out_ch, in_size, K, n_layers=5):
        super().__init__()

        self.K = K
        self.conv = utm.CNN_nopool(in_ch, inter_ch, K, n_layers)
        self.phi = nn.Linear((in_size//2)**2, out_ch)
    
    def forward(self, x):
        out = self.conv(x)
        return self.phi(out)

class SimpleEncoder(nn.Module):
    """
    Simple encoder. Adds an mlp to the output of the PureCNNEncoder.
    """
    def __init__(self, in_ch, inter_ch, out_ch, in_size, K, stop=0, 
                 num_mlp_layers=3):
        super().__init__()

        self.K = K

        self.conv = utm.MaxpoolEncoder(
            in_ch, inter_ch, out_ch * K, in_size, stop=stop)
        
        self.mlp = utm.MLP([out_ch] * num_mlp_layers)
        # normalization layer ?

    def forward(self, x):
        z = self.conv(x)
        # format by slots
        zs = torch.stack(z.chunk(self.K, -1), 1)
        zs = self.mlp(zs)
        return zs

### Slot-Memory

class SelfAttentionLayer(nn.Module):
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

class AttentionLayerSparse(nn.Module):
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
        aw = qs.view(-1, 1, self.Fqk) @ ks.view(-1, self.Fqk, 1)
        aw = aw.squeeze()
        # softmax reduction
        aw = scatter_softmax(aw, src)

        # out = aw.view([-1, H, 1]) * vs
        # out = scatter_sum(out, src)
        # out = out.reshape([-1, self.Fv])

        return out

class TransformerBlock(nn.Module):
    """
    Implements a full Transformer block, with skip connexions, layernorm
    and an mlp.

    Arguments:
        - d: dimension of a head
        - projdim: dimension for projection inside mhsa
        - h: number of heads
    """
    def __init__(self, d, projdim, h):
        super().__init__()

        self.d = d
        self.h = h
        self.projdim = projdim

        if self.projdim != self.d:
            # need an additional linear layer for the skip-connexion
            # to ajust the dimensionality of outputs
            self.proj = nn.Linear(projdim, d)

        self.norm1 = nn.LayerNorm([d])
        self.norm2 = nn.LayerNorm([d])

        self.mhsa = SelfAttentionLayer(d, projdim, projdim, h)
        # TODO: check papers for hparams
        self.mlp = utm.MLP([d, projdim, d])

    def forward(self, x):

        y = self.mhsa(x)
        if self.projdim != self.d:
            # ajust dimension of output
            y = self.proj(y)
        y = self.norm1(x + y)

        z = self.mlp(y)
        z = self.norm2(y + z)

        return z

### Slot-Memory architectures

class SlotMem(nn.Module):
    """
    A GNN where the edge model + edge aggreg is a self-attention layer.
    There are K hidden states and cells, each corresponding to a particular
    memory slot. The LSTM parameters are shared between all slots.

    Dense implem (should we call this a GNN ?)

    We have two choices for what vectors we use for the self-attention update:
    hidden vectors of cells. We'll use cells here, but that may not be the best
    choice.

    The model only does one forward pass on the sequence.

    Arguments:
        - B: batch size, must be specified in advance;
        - K: number of memory slots;
        - Fin: number of features of the input;
        - Fmem: number of features of each memory slot;
        - H: number of dims of the projections in Transformer;
        - nheads: number of heads in the self-attention mechanism;
        - gating: can one of "slot" or "feature".
            "slot" means the gating mechanism happens at the level of the whole
            slot; 
            "feature" means the gating mechanism happens at the level of
            individual features.
    """
    def __init__(self, K, Fmem, H, nheads, gating="feature"):
        super().__init__()

        self.K = K
        self.Fmem = Fmem
        self.H = H
        self.nheads = nheads
        self.gating = gating

        # maybe replace with something else
        self.self_attention = TransformerBlock(
            Fmem,
            H,
            nheads,
        )

        # define the gating net
        if gating == "feature":
            self.proj = nn.Linear(2 * Fmem, 2 * Fmem)
        elif gating == "slot":
            self.proj = nn.Linear(2 * Fmem, 2)
        else:
            raise ValueError("the 'gating' argument must be one of:\n"
                             "\t- 'slot'\n\t- 'feature'")

    def _mem_init(self, bsize):
        """
        Some form of initialization where the vectors are unique.

        The batch size must be provided.
        """
        memory0 = torch.cat([
            torch.eye(self.K).expand([bsize, self.K, self.K]),
            torch.zeros([bsize, self.K, self.Fmem - self.K])
        ], -1)
        return memory0

    def forward(self, x, memory):
        # x can also be None when no output is provided

        # add input vectors to perform self-attention
        if x is not None:
            mem_cat = torch.cat([memory, x], 1)
        else:
            mem_cat = memory
        # candidate input
        mem_update = self.self_attention(mem_cat)[:, :self.K]

        # compute forget and input gates
        f, i = self.proj(torch.cat([memory, mem_update], -1)).chunk(2, -1)

        # update memory
        # this mechanism may be refined
        memory = memory * torch.sigmoid(f) + mem_update * torch.sigmoid(i)

        # for now the output is the memory
        output = memory

        return output, memory

class SlotMemIndependent(nn.Module):
    """
    Slot-memory, LSTM structure, no interaction between slots.

    The dimensions of input and outputs can differ from the dimension
    of memory.
    """
    def __init__(self, K, Fmem, Fin, Fout):
        super().__init__()

        self.K = K
        self.Fmem = Fmem
        self.Fin = Fin

        self.proj = nn.Linear(Fin + Fmem, 4 * Fmem)
        self.out_proj = nn.Linear(Fmem, Fout)

    def _mem_init(self, bsize):
        """
        Initializes hidden state and cell.
        """
        h0 = torch.cat([
            torch.eye(self.K).expand([bsize, self.K, self.K]),
            torch.zeros([bsize, self.K, self.Fmem - self.K])
        ], -1)
        c0 = torch.cat([
            torch.eye(self.K).expand([bsize, self.K, self.K]),
            torch.zeros([bsize, self.K, self.Fmem - self.K])
        ], -1)
        
        return torch.cat([h0, c0], -1)

    def forward(self, x, mem):
        h, c = mem.chunk(2, -1)
        
        f, i, c_tilde, o = self.proj(torch.cat([x, h], -1)).chunk(4, -1)
        c_tilde = torch.tanh(c_tilde)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * c_tilde
        h = torch.sigmoid(o) * c
        
        out = self.out_proj(h)
        mem = torch.cat([h, c], -1)
        
        return out, mem

### Slot-distance functions

class L2Dist(nn.Module):
    """
    Simple slot-wise L2 distance.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z, m):
        return ((z - m)**2).sum(-1)**.5

class NegativeCosSim(nn.Module):
    """
    Slot-wise negative cosine similarity.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z, m):
        # z, m :: [B, N, F]
        # TODO check this
        B, N, F = z.shape
        normz = (z**2).sum(-1).unsqueeze(-1)
        normm = (m**2).sum(-1).unsqueeze(-1)

        dot = z.view([B, N, 1, F]) @ m.view([B, N, F, 1])
        return dot.squeeze(-1) / (normz * normm)

class MatchDistance(nn.Module):
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
        super().__init__()

        # note: no heads, maybe add ?
        # sparse implem for dealing with sparse edges ?
        self.Fin = Fin
        self.Fqk = Fqk
        self.Ftot = 2 * Fqk

        self.hard = hard
        # temperature for the gumbel-softmax
        self.tau = tau

        self.attention = AttentionLayerSparse(Fin=Fin, Fqk=Fqk)

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

class BaseCompleteModel(nn.Module):
    """
    Base class for a complete model, with an encoder, a slot-memory mechanism
    and a distance mechanism. Subclasses of this may define the models in the
    __init__ function directly.
    """    
    def __init__(self, C_phi, M_psi, Delta_xi, model_diff=False):
        super().__init__()

        self.model_diff = model_diff

        self.C_phi = C_phi
        self.M_psi = M_psi
        self.Delta_xi = Delta_xi

    def next(self, x):
        return self.M_psi(self.C_phi(x))

    def mem_init(self, bsize):
        return self.M_psi._mem_init(bsize)

    def forward(self, x1, x2, mem=None, n_recurrent_passes=1):
        """
        Computes the energy between x1 and x2. Usually, x1 is some state and
        x2 is some state in the future. 

        n_recurrent_passes denotes the number of internal recurrent steps done
        by the model before comparing x1 and x2. No input is given to the
        model.
        """
        if mem is None:
            mem = self.mem_init(x1.shape[0])

        z1 = self.C_phi(x1)
        z2 = self.C_phi(x2)

        # z1/z2 :: [B, K, F]

        out, next_mem = self.M_psi(z1, mem)

        for _ in range(n_recurrent_passes-1):
            # do additional recurrent passes with no input
            out, next_mem = self.M_psi(None, next_mem)

        # compute distance/energy d
        if not self.model_diff:
            d = self.Delta_xi(z2, out)
        else:
            # alternative: model the difference
            d = self.Delta_xi(z2, z1 + out)

        return d, next_mem

    def forward_seq(self, xs, mem=None):
        """
        Computes the forward pass on the given sequence xs.
        This is done by comparing the neighbouring elements one by one.

        The sequence xs is expected as a tensor of size sequence_length, bsize,
        etc ...

        returns the sequence of distances, the sequences of encoded hidden
        representations and the sequence of predicted hidden representations.
        """
        d_list = []
        z_list = []
        z_hat_list = []

        if mem is None:
            mem = self.mem_init(xs.shape[1])

        for i in range(len(xs) - 1):
            z1 = self.C_phi(xs[i])
            z2 = self.C_phi(xs[i+1])

            out, mem = self.M_psi(z1, mem)

            if not self.model_diff:
                z2_hat = out
            else:
                z2_hat = z1 + out
    
            d = self.Delta_xi(z2, z2_hat)

            d_list.append(d)
            z_list.append(z2)
            z_hat_list.append(z2_hat)

        return d_list, z_list, z_hat_list

    # def forward_rollout(self, x, xs):
    #     """
    #     Computes the energy between x and a list of inputs xs.
    #     The energy between x and xs[i] is computed by doing i recurrent
    #     passes without input after encoding x, and comparing to the encoding
    #     of x[i].
    #     Returns a list of energies.
    #     """
    #     # TODO: modify for tensor inputs
    #     raise NotImplementedError

    #     # TODO: modify this to comply ith new slotmem interface
    #     if not isinstance(xs, list):
    #         xs = [xs]

    #     z = self.C_phi(x)
    #     # TODO paralellize the following
    #     zs = [self.C_phi(y) for y in xs]

    #     L = len(xs)
    #     mlist = [self.M_psi(z)]

    #     for _ in range(L-1):
    #         mlist.append(self.M_psi())

    #     if not self.model_diff:
    #         d = [self.Delta_xi(zz, m) for zz, m in zip(zs, mlist)]
    #     else:
    #         # model the difference
    #         # first compute the cumulative sum of elements of mlist
    #         mtensor = torch.stack(mlist, 0)
    #         cumsum_mtensor = mtensor.cumsum()
    #         # then model the transitions
    #         d = [self.Delta_xi(zz, z + m) for zz, m in zip(zs, cumsum_mtensor)]

    #     return d

class CompleteModel_Debug(BaseCompleteModel):
    """
    Simpler, debugging version.
    """
    def __init__(self, K, Fmem, hidden_dim, input_dims, nheads,
                 model_diff=True):
        self.H, self.W, self.C = input_dims
        
        C_phi = PureCNNEncoder_nopool(3, 32, Fmem, self.H, K)
        M_psi = SlotMemIndependent(K, hidden_dim, Fmem, Fmem)
        Delta_xi = L2Dist()

        super().__init__(C_phi, M_psi, Delta_xi, model_diff=model_diff)

class CompleteModel_SlotDistance(BaseCompleteModel):
    """
    Slot-wise distance fn.
    """
    def __init__(self, K, Fmem, hidden_dim, input_dims, nheads, 
                 model_diff=False):

        self.H, self.W, self.C = input_dims
        
        C_phi = PureCNNEncoder_nopool(3, 32, Fmem, self.H, K)
        M_psi = SlotMem(K, Fmem, hidden_dim, nheads)
        Delta_xi = L2Dist()

        super().__init__(C_phi, M_psi, Delta_xi, model_diff=model_diff)

class CompleteModel_SoftMatchingDistance(BaseCompleteModel):
    """
    Simple CNN encoder;
    Parallel-LSTM with dot-product-attention communication between slots;
    Soft-slot-matching distance function.
    """
    def __init__(self, K, Fmem, hidden_dim, input_dims, nheads):

        self.H, self.W, self.C = input_dims

        C_phi = SimpleEncoder(3, 32, Fmem, self.H, K)
        M_psi = SlotMem(K, Fmem, hidden_dim, nheads)
        Delta_xi = MatchDistance(Fmem, Fmem, hard=False)

        super().__init__(C_phi, M_psi, Delta_xi)

class CompleteModel_HardMatchingDistance(BaseCompleteModel):
    """
    Simple CNN encoder;
    Parallel-LSTM with dot-product-attention communication between slots;
    Hard-slot-matching distance function.
    """
    def __init__(self, K, Fmem, hidden_dim, input_dims, nheads):

        self.H, self.W, self.C = input_dims

        C_phi = SimpleEncoder(3, 32, Fmem, self.H, K)
        M_psi = SlotMem(K, Fmem, hidden_dim, nheads)
        Delta_xi = MatchDistance(Fmem, Fmem, hard=True)

        super().__init__(C_phi, M_psi, Delta_xi)

### For processing sequences

def recurrent_apply(recurrent_model, seq, mem0):
    # S :: [s, b] + input_dims
    # mem0 is initial memory

    out_list = []
    mem = mem0

    for i in range(len(seq) - 1):
        s1 = seq[i]
        s2 = seq[i+1]

        d, mem = recurrent_model(s1, s2, mem)
        out_list += [d]

    return torch.cat(out_list, 0)

def recurrent_apply_contrastive(recurrent_model, seq, mem0=None):
    """
    Same as recurrent_apply, applies a recurrent model on a sequence of inputs,
    but also computes the time-contrastive term by sampling arbitrary 
    next-states.

    One-hop prediction.

    mem0 is first memory.
    """
    if mem0 is None:
        B = seq.shape[1]
        mem0 = recurrent_model.mem_init(B)
    N = len(seq)
    # compute randomly shuffled sequence
    rand = (torch.randint(1, N-1, (N-1,)) + torch.arange(N-1)).fmod(N-1)

    normal_range = list(range(N-1))
    random_range = rand.tolist()

    out_list = []
    out_list_contrastive = []

    mem = mem0

    for i, j in zip(normal_range, random_range):
        s1 = seq[i]
        s2 = seq[i+1]
        sc = seq[j+1]

        d, _ = recurrent_model(s1, s2, mem)
        d_contrastive, mem = recurrent_model(s1, sc, mem)

        out_list += [d]
        out_list_contrastive += [d_contrastive]

    return torch.stack(out_list, 0), torch.stack(out_list_contrastive, 0)

def recurrent_apply_contrastive_Lsteps(recurrent_model, seq, L):
    """
    Same as before, but the predictions are rolled-out on L steps and the loss
    is computed between all the predicted steps.

    TODO: Maybe only rollout from a random subset of the start states ?
    TODO: How to compute contrastive samples ? For now, completely random
          sequence.
    """
    raise NotImplementedError
    # TODO: adapt this to new SlotMem interface

    N = len(seq)
    assert(N > L, (f"Length of the rollout ({L}) should be strictly smaller"
                   f" than length of the sequence ({N})"))
    # set of random sequences
    # check it is correct
    rand = (torch.randint(1, N-L, (N-L, L)) + torch.arange(N-L)).fmod(N-L)
    random_range_list = rand.tolist()
    normal_range = range(L)

    for t0, random_range in range(random_range_list):
        # start state loop
        t = t0
        s = seq[t]

        for i, j in zip(normal_range, random_range):
            # sequence length loop
            strue = seq[t+i]
            scontrastive = seq[t+j]
            pred = recurrent_model(s)
            # TODO: finish this

### Tests

# model = SlotMem(7, 4, 10, 2)
# x = torch.rand(7, 4, 10)