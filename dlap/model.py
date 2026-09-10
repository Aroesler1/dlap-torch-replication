"""
PyTorch port of `FeedForwardModelWithNA_GAN` (Chen, Pelger & Zhu).

Conventions follow the authors' TF code exactly:
    w_{t,i}  = FFN( I_{t,i}, h_t )                       h_t = LSTM(macro_{1..t})
    M_{t+1}  = 1 + sum_i w_{t,i} R^e_{t+1,i}             (their "SDF"; the SDF *factor* is F = 1 - M = -w'R)
    loss(g)  = mean_{j,i}  T_i/max(T) * ( 1/T_i sum_t  M_{t+1} R^e_{t+1,i} g_{j,t,i} )^2
where g = 1 for the unconditional loss and g = tanh(FFN(I, h^g)) (8 outputs) for the adversary.

Extensions (all OFF by default, i.e. plain replication):
    turnover_penalty  : + lambda * mean_t sum_i |w~_{t,i} - w~_{t-1,i}|  (w~ = L1-normalised weights)
    sdf_hinge         : + lambda * mean_t max(0, -M_t)^2   (admissibility: SDF should be positive)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden, out_dim, out_act=None):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    if out_act is not None:
        layers.append(out_act)
    return nn.ModuleList(layers)


class MacroLSTM(nn.Module):
    """LSTM over the whole macro history -> low-dimensional hidden state per month."""

    def __init__(self, in_dim, units, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, units, num_layers=num_layers, batch_first=True)

    def forward(self, macro_seq, drop_p=0.0):
        x = F.dropout(macro_seq, drop_p, self.training) if drop_p > 0 else macro_seq
        out, _ = self.lstm(x.unsqueeze(0))
        return out.squeeze(0)               # [T_seq, units]


class SDFNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.C = cfg['individual_feature_dim']
        self.K = cfg['macro_feature_dim']
        self.use_rnn = bool(cfg.get('use_rnn', True)) and self.K > 0
        if self.use_rnn:
            self.rnn = MacroLSTM(self.K, cfg['num_units_rnn'][0], cfg.get('num_layers_rnn', 1))
            state_dim = cfg['num_units_rnn'][0]
        else:
            state_dim = self.K
        self.ffn = _mlp(self.C + state_dim, cfg['hidden_dim'][:cfg['num_layers']], 1)
        self.drop_p = 1.0 - cfg.get('dropout', 1.0)      # authors store *keep* prob

    def macro_state(self, macro_seq):
        if self.K == 0:
            return None
        if self.use_rnn:
            return self.rnn(macro_seq, self.drop_p if self.training else 0.0)
        return macro_seq

    def weights(self, I, mask, state, t0):
        """
        I: [T, N, C], mask: [T, N] bool, state: [T_seq, S] (LSTM output over the full history) or None.
        t0: index in state where this split starts.  Returns flat w for masked (t,i) pairs (row-major).
        """
        T = I.shape[0]
        x = I[mask]                                            # [M, C]
        if state is not None:
            st = state[t0:t0 + T]                              # [T, S]
            idx_t = torch.nonzero(mask, as_tuple=True)[0]      # month index of each masked row
            x = torch.cat([x, st[idx_t]], dim=1)
        h = x
        for layer in self.ffn:
            h = layer(h)
            if isinstance(layer, nn.ReLU) and self.drop_p > 0:
                h = F.dropout(h, self.drop_p, self.training)
        return h.squeeze(-1)                                   # [M]

    @staticmethod
    def sdf_from_w(w_flat, R, mask, normalize_w=False):
        """M_t = 1 + sum_i w_{t,i} R_{t,i}.  Returns (M [T], w dense [T, N])."""
        W = torch.zeros_like(R)
        W[mask] = w_flat
        M = 1.0 + (W * R).sum(1)
        if normalize_w:                                        # authors' optional N-normalisation
            N_t = mask.sum(1).float()
            M = 1.0 + (W * R).sum(1) / N_t * N_t.mean()
        return M, W

    def forward(self, I, R, mask, state, t0):
        w = self.weights(I, mask, state, t0)
        M, W = self.sdf_from_w(w, R, mask, self.cfg.get('normalize_w', False))
        return w, W, M


class MomentNet(nn.Module):
    """Adversary: g_{t,i} in R^D built from (I_{t,i}, h^g_t), h^g = LSTM_g(macro)."""

    def __init__(self, cfg):
        super().__init__()
        self.C = cfg['individual_feature_dim']
        self.K = cfg['macro_feature_dim']
        self.use_rnn = bool(cfg.get('use_rnn', True)) and self.K > 0
        if self.use_rnn:
            self.rnn = MacroLSTM(self.K, cfg['num_units_rnn_moment'][0], cfg.get('num_layers_rnn_moment', 1))
            state_dim = cfg['num_units_rnn_moment'][0]
        else:
            state_dim = self.K
        self.D = cfg['num_condition_moment']
        self.ffn = _mlp(self.C + state_dim, cfg.get('hidden_dim_moment', [])[:cfg.get('num_layers_moment', 0)],
                        self.D, out_act=nn.Tanh())
        self.drop_p = 1.0 - cfg.get('dropout', 1.0)

    def forward(self, I, mask, macro_seq, t0):
        T, N, _ = I.shape
        x = I[mask]
        if self.K > 0:
            st = self.rnn(macro_seq, self.drop_p if self.training else 0.0) if self.use_rnn else macro_seq
            st = st[t0:t0 + T]
            idx_t = torch.nonzero(mask, as_tuple=True)[0]
            x = torch.cat([x, st[idx_t]], dim=1)
        h = x
        for layer in self.ffn:
            h = layer(h)
            if isinstance(layer, nn.ReLU) and self.drop_p > 0:
                h = F.dropout(h, self.drop_p, self.training)
        G = torch.zeros(T, N, self.D, device=I.device, dtype=I.dtype)
        G[mask] = h
        return G.permute(2, 0, 1)                               # [D, T, N]


# ------------------------------------------------------------------ losses -----------------
def pricing_loss(M, R, mask, G, T_i, weighted=True):
    """
    M: [T], R: [T,N], mask: [T,N], G: [D,T,N] (or None for unconditional), T_i: [N] months per stock.
    Mirrors `_add_loss` in the authors' code (including the T_i / max(T_i) weighting).
    """
    RM = R * mask.float()
    x = RM * M[:, None]                                        # [T, N]
    if G is None:
        emp = x.sum(0) / T_i.clamp(min=1)                      # [N]
    else:
        emp = (x[None] * G).sum(1) / T_i.clamp(min=1)[None]    # [D, N]
    sq = emp ** 2
    if weighted:
        sq = sq * (T_i / T_i.max())[None] if G is not None else sq * (T_i / T_i.max())
    return sq.mean()


def turnover_penalty(W, mask):
    """mean over months of sum_i |w~_t,i - w~_{t-1,i}| using L1-normalised weights (same PERMNO index)."""
    Wn = W / (W.abs().sum(1, keepdim=True) + 1e-12)
    both = (mask[1:] & mask[:-1]).float()
    return ((Wn[1:] - Wn[:-1]).abs() * both).sum(1).mean()


def sdf_hinge(M):
    return (F.relu(-M) ** 2).mean()
