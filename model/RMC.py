"""
Defines the RMC module.
"""

class SelfAttentionLayer(torch.nn.Module):
    """
    Multi-head Self-Attention layer.

    TODO: add support for diffrent numbers of objects between queries and
        keys/values.

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
        # alternative formulation of forward pass

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

### RMC

class RMC(torch.nn.Module):
    """
    Relational Memory Core.

    TODO: test the LSTM-like update

    Please note that using this model with multiple input vectors, in LSTM
    mode leads to concatenating the inputs in the first dimension when
    computing the gates for the LSTM update. This seems suboptimal from a
    relational architecture point of view, and would orient us towards using
    a per-slot LSTM (where the gates would be computed per-slot) instead.

    -> It breaks the permutation invariance of inputs
    """
    modes = ['RNN', 'LSTM']

    def __init__(self,
                 N,
                 d,
                 h,
                 b,
                 Nx=None,
                 mode='RNN',
                 device=torch.device('cpu')):
        super().__init__()
        
        # TODO: do we need the batch size in advance ?
        self.N = N # number of slots
        if Nx is None: 
            self.Nx = N # number of slots for the input
        else:
            self.Nx = Nx
        self.d = d # dimension of a single head
        self.h = h # number of heads
        self.b = b # batch size

        self.device = device
        self.mode = mode

        # initialize memory M
        # TODO: unique initialization of each slot
        M = torch.zeros([self.b, self.N, self.d * self.h])
        self.register_buffer('M', M)

        # modules
        self.self_attention = TransformerBlock(h * d, h)

        if mode in ['LSTM', 'LSTM_noout']:
            # hidden state
            hid = torch.zeros([self.b, self.N, self.d * self.h])
            self.register_buffer('hid', hid)

            # scalar LSTM gates
            self.Wf = Linear(d * h * self.Nx, 1)
            self.Uf = Linear(d * h, 1)

            self.Wi = Linear(d * h * self.Nx, 1)
            self.Ui = Linear(d * h, 1)

            self.Wo = Linear(d * h * self.Nx, 1)
            self.Uo = Linear(d * h, 1)

    def _forwardRNN(self, x):
        # vanilla recurrent pass
        # x :: [b, N, f]

        M_cat = torch.cat([self.M, x], 1)
        M = self.self_attention(M_cat)[:, :self.N]

        self.register_buffer('M', M)

        # output is flattened memory
        out = M.view(self.b, -1)
        return out

    def _forwardLSTM(self, x):
        # LSTM recurrent pass
        M_cat = torch.cat([self.M, x], 1)
        Mtilde = self.self_attention(M_cat)[:, :self.N]

        x_cat = x.flatten(1).unsqueeze(1)

        f = self.Wf(x_cat) + self.Uf(self.M)
        i = self.Wi(x_cat) + self.Uf(self.M)
        o = self.Wo(x_cat) + self.Uo(self.M)

        M = torch.sigmoid(f) * self.M + torch.sigmoid(i) * torch.tanh(Mtilde)
        hid = torch.sigmoid(o) * torch.tanh(M)

        self.register_buffer('M', M)
        self.register_buffer('hid', hid)

        return hid.view(self.b, -1)

    def _forwardLSTM_noout(self, x):
        # LSTM recurrent pass, no output gate
        M_cat = torch.cat([self.M, x], 1)
        Mtilde = self.self_attention(M_cat)[:, :self.N]
        
        x_cat = x.flatten(1).unsqueeze(1)

        f = self.Wf(x_cat) + self.Uf(self.M)
        i = self.Wi(x_cat) + self.Uf(self.M)
        o = self.Wo(x_cat) + self.Uo(self.M)

        M = torch.sigmoid(f) * M + torch.sigmoid(i) * torch.tanh(Mtilde)
        hid = M

        self.register_buffer('M', M)
        self.register_buffer('hid', hid)

        return hid.view(self.b, -1)

    def forward(self, x):
        if self.mode == 'RNN':
            return self._forwardRNN(x)
        elif self.mode == 'LSTM':
            return self._forwardLSTM(x)


