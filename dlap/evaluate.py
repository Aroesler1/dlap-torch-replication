import os, json, copy
import numpy as np
import torch
import torch.nn as nn


def sharpe_monthly(f):
    f = np.asarray(f, dtype=np.float64)
    return float(f.mean() / f.std()) if f.std() > 0 else 0.0


def max_drawdown(f):
    """max drawdown of cumulative (arithmetic) return of the factor"""
    c = np.cumsum(f); peak = np.maximum.accumulate(c)
    return float((peak - c).max())


def normalize_w(W, mask):
    """L1-normalise weights each month (authors' `normalized=True`)."""
    Wn = np.where(mask, W, 0.0)
    s = np.abs(Wn).sum(1, keepdims=True); s[s == 0] = 1.0
    return Wn / s


def sdf_factor_from_w(Wn, R, mask):
    """F_t = -(sum_i w~_{t,i} R_{t,i})  =  1 - M_t"""
    return -(np.where(mask, Wn * R, 0.0)).sum(1)


def ev_xsr2(w_dense, R, mask):
    """
    Authors' `calculateStatistics`: project each month's returns on the loading vector (w or beta):
    R_hat_t = (w_t'R_t / w_t'w_t) w_t.   Returns (EV, XS-R2, weighted XS-R2).
    """
    R = np.where(mask, R, 0.0); W = np.where(mask, w_dense, 0.0)
    num = (W * R).sum(1); den = (W * W).sum(1); den[den == 0] = 1.0
    R_hat = (num / den)[:, None] * W
    resid = np.where(mask, R - R_hat, 0.0)
    T_i = mask.sum(0).astype(float); N_t = mask.sum(1).astype(float)
    ok = T_i > 0
    ev = 1 - np.mean((resid ** 2).sum(1) / N_t) / np.mean((R ** 2).sum(1) / N_t)
    mres = resid.sum(0)[ok] / T_i[ok]; mR = R.sum(0)[ok] / T_i[ok]
    xs = 1 - np.mean(mres ** 2) / np.mean(mR ** 2)
    xsw = 1 - np.mean(mres ** 2 * T_i[ok]) / np.mean(mR ** 2 * T_i[ok])
    return float(ev), float(xs), float(xsw)


def turnover(Wn, mask):
    """
    Monthly turnover of the L1-normalised book: sum_i |w~_t,i - w~_{t-1,i}| (same PERMNO index).
    Positions that disappear (stock leaves the sample) are counted as fully liquidated.
    """
    d = np.abs(Wn[1:] - Wn[:-1]).sum(1)
    return d                                                   # [T-1]


def net_of_cost_factor(F, to, cost_bps):
    """subtract one-way cost * turnover from each month's factor return (F is L1-normalised so cost is per $1 gross)"""
    F = np.asarray(F, dtype=np.float64).copy()
    F[1:] -= (cost_bps / 1e4) * to
    return F


def yearly_sharpe(F, dates):
    yrs = np.asarray(dates) // 100
    out = {}
    for y in np.unique(yrs):
        f = F[yrs == y]
        out[int(y)] = sharpe_monthly(f) * np.sqrt(12)
    return out


# ---------------------------------------------------------------- ensemble ------------------
@torch.no_grad()
def ensemble_weights(cfg, ckpt_paths, bundles, which='sdf_best_sharpe', device='cuda'):
    """Average raw weights across seeds (authors: mean of raw w, then L1-normalise)."""
    from .model import SDFNet
    out = []
    for b in bundles:
        out.append(np.zeros((b.T, b.I.shape[1]), dtype=np.float64))
    for p in ckpt_paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        sd = ck[which] if ck[which] is not None else ck['sdf_last']
        net = SDFNet(cfg).to(device); net.load_state_dict(sd); net.eval()
        for k, b in enumerate(bundles):
            state = net.macro_state(b.macro_seq)
            w, W, M = net(b.I, b.R, b.mask, state, b.t0)
            out[k] += W.cpu().numpy() / len(ckpt_paths)
    return out


@torch.no_grad()
def macro_states(cfg, ckpt_path, bundle, device='cuda'):
    from .model import SDFNet
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = SDFNet(cfg).to(device); net.load_state_dict(ck['sdf_best_sharpe'] or ck['sdf_last']); net.eval()
    st = net.macro_state(bundle.macro_seq)
    return None if st is None else st[bundle.t0:bundle.t0 + bundle.T].cpu().numpy()


# ---------------------------------------------------------------- beta network --------------
class BetaNet(nn.Module):
    """Second-stage FFN from the authors' `create_RF_data.py` + `model_RtnFcst.py`: predict R_{t,i} F_t (x50)."""

    def __init__(self, C, hidden=(32, 16, 8), keep=0.95):
        super().__init__()
        layers, d = [], C
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(1 - keep)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_beta(splits, F_by_split, epochs=2048, lr=1e-3, seeds=(0,), device='cuda', log=print, scale=50.0):
    """
    Train beta(I) to predict R*F*50 on train, select by valid MSE, ensemble over seeds.
    Returns list of dense beta arrays [T,N] for (train, valid, test).
    """
    tr, va, te = splits
    X = [torch.tensor(s.I[s.mask], device=device) for s in splits]
    Y = [torch.tensor((s.R * F[:, None] * scale)[s.mask], device=device, dtype=torch.float32)
         for s, F in zip(splits, F_by_split)]
    preds = [np.zeros((s.T, s.N)) for s in splits]
    for seed in seeds:
        torch.manual_seed(seed)
        net = BetaNet(tr.C).to(device); opt = torch.optim.Adam(net.parameters(), lr=lr)
        best, best_sd = float('inf'), None
        for ep in range(epochs):
            net.train(); loss = ((net(X[0]) - Y[0]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 16 == 0:
                net.eval()
                with torch.no_grad():
                    lv = ((net(X[1]) - Y[1]) ** 2).mean().item()
                if lv < best:
                    best, best_sd = lv, copy.deepcopy(net.state_dict())
        net.load_state_dict(best_sd); net.eval()
        with torch.no_grad():
            for k, s in enumerate(splits):
                p = np.zeros((s.T, s.N)); p[s.mask] = net(X[k]).cpu().numpy(); preds[k] += p / len(seeds)
        log(f'   beta net seed {seed}: best valid MSE {best:.4f}')
    return preds


def decile_portfolios(beta, R, mask, deciles=10):
    """equal-weighted decile portfolios sorted on beta each month -> [T, deciles] mean returns"""
    T = R.shape[0]; out = np.full((T, deciles), np.nan)
    for t in range(T):
        m = mask[t]; b = beta[t][m]; r = R[t][m]
        if len(b) < deciles:
            continue
        q = np.floor(np.argsort(np.argsort(b)) * deciles / len(b)).astype(int)
        for d in range(deciles):
            out[t, d] = r[q == d].mean()
    return out
