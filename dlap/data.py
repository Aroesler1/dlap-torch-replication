"""
Data layer for Chen–Pelger–Zhu "Deep Learning in Asset Pricing" replication.

Reads the *exact* npz format shipped by the authors (Google-Drive `datasets/` folder):

  datasets/char/Char_{train,valid,test}.npz
      data     : float array [T, N, 1 + 46]   -> [:, :, 0] = excess return, [:, :, 1:] = 46 characteristics
                 characteristics are rank-normalised to [-0.5, 0.5]; missing = -99.99 in the return column
      date     : [T]  yyyymm
      variable : [47] names (first is the return)
  datasets/macro/macro_{train,valid,test}.npz
      data     : float array [T, K]  (K = 178 macro series)
      variable : [K] names

Stock index i is the same PERMNO across all t inside one split, which is what lets us compute turnover.
"""
import numpy as np

UNK = -99.99

CHAR_CATEGORIES = {
    'Past Returns':      ['r2_1', 'r12_2', 'r12_7', 'r36_13', 'ST_REV', 'LT_Rev'],
    'Investment':        ['Investment', 'NOA', 'DPI2A', 'NI'],
    'Profitability':     ['PROF', 'ATO', 'CTO', 'FC2Y', 'OP', 'PM', 'RNA', 'ROA', 'ROE', 'SGA2S', 'D2A'],
    'Intangibles':       ['AC', 'OA', 'OL', 'PCM'],
    'Value':             ['A2ME', 'BEME', 'C', 'CF', 'CF2P', 'D2P', 'E2P', 'Q', 'S2P', 'Lev'],
    'Trading Frictions': ['AT', 'Beta', 'IdioVol', 'LME', 'LTurnover', 'MktBeta', 'Rel2High', 'Resid_Var', 'Spread', 'SUV', 'Variance'],
}
VAR2CAT = {v: k for k, vs in CHAR_CATEGORIES.items() for v in vs}


class Split:
    """One of train / valid / test."""

    def __init__(self, char_path, macro_path=None, macro_idx=None, macro_mean=None, macro_std=None):
        tmp = np.load(char_path, allow_pickle=True)
        data = tmp['data'].astype(np.float32)
        self.R = data[:, :, 0]                       # [T, N]
        self.I = data[:, :, 1:]                      # [T, N, 46]
        self.mask = self.R != UNK                    # [T, N]
        self.R = np.where(self.mask, self.R, 0.0).astype(np.float32)
        self.dates = np.array(tmp['date'])
        if np.issubdtype(self.dates.dtype, np.number) and self.dates.max() > 999999:
            self.dates = self.dates // 100        # authors' files store yyyymmdd; the rest of the code expects yyyymm
        self.char_names = [str(v) for v in tmp['variable'][1:]]
        self.T, self.N, self.C = self.I.shape

        if macro_path is None:
            self.M = np.zeros((self.T, 0), dtype=np.float32)
            self.macro_names = []
            self.macro_mean, self.macro_std = None, None
        else:
            tm = np.load(macro_path, allow_pickle=True)
            md = np.asarray(tm['data'], dtype=np.float64)   # authors standardise in float64 and feed float32 placeholders
            names = [str(v) for v in tm['variable']]
            if macro_idx is None or macro_idx == 'all':
                idx = np.arange(md.shape[1])
            elif macro_idx == '178':          # authors' convention on a 338-column macro file
                idx = np.sort(np.concatenate((np.arange(124), np.arange(284, 338))))
            else:
                idx = np.sort(np.array(macro_idx, dtype=int))
            md = md[:, idx]
            self.macro_names = [names[i] for i in idx]
            if macro_mean is None:            # normalise with *training* moments only (no look-ahead)
                macro_mean, macro_std = md.mean(0), md.std(0)
                macro_std = np.where(macro_std == 0, 1.0, macro_std)
            self.macro_mean, self.macro_std = macro_mean, macro_std
            self.M = ((md - macro_mean) / macro_std).astype(np.float32)
        self.K = self.M.shape[1]

    # --- helpers used by the loss --------------------------------------------------------
    def T_i(self):
        """number of months each stock is observed (loss weights)"""
        return self.mask.sum(0)

    def N_t(self):
        return self.mask.sum(1)

    def summary(self, name=''):
        return (f"{name:6s} T={self.T:4d} months ({self.dates[0]}–{self.dates[-1]})  "
                f"N={self.N:6d} permnos  obs={int(self.mask.sum()):8d}  "
                f"avg stocks/month={self.mask.sum(1).mean():7.1f}  chars={self.C}  macro={self.K}")


def load_all(cfg):
    """Load train / valid / test with macro normalised on the training window."""
    macro_idx = cfg.get('macro_idx', None)
    use_macro = cfg.get('macro_feature_dim', 0) > 0 and cfg.get('macro_feature_file') is not None
    tr = Split(cfg['individual_feature_file'],
               cfg['macro_feature_file'] if use_macro else None, macro_idx)
    va = Split(cfg['individual_feature_file_valid'],
               cfg['macro_feature_file_valid'] if use_macro else None, macro_idx, tr.macro_mean, tr.macro_std)
    te = Split(cfg['individual_feature_file_test'],
               cfg['macro_feature_file_test'] if use_macro else None, macro_idx, tr.macro_mean, tr.macro_std)
    return tr, va, te


# --------------------------------------------------------------------------------------------
# Synthetic data in the authors' format – used only to smoke-test the pipeline without CRSP data.
# A planted, mildly non-linear SDF makes the "right" answer recoverable so we can sanity-check.
# --------------------------------------------------------------------------------------------
def make_synthetic(out_dir, T=(60, 24, 48), N=400, C=46, K=20, seed=0, missing=0.15):
    import os
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, 'char'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'macro'), exist_ok=True)
    names = [v for vs in CHAR_CATEGORIES.values() for v in vs][:C]
    while len(names) < C:
        names.append(f'x{len(names)}')
    Ttot = sum(T)
    # a slow "business cycle" hidden state drives the price of risk
    state = np.zeros(Ttot)
    for t in range(1, Ttot):
        state[t] = 0.9 * state[t - 1] + 0.3 * rng.standard_normal()
    macro = state[:, None] * rng.standard_normal((1, K)) + rng.standard_normal((Ttot, K))
    macro = np.cumsum(macro, axis=0) * 0.1 + macro          # some non-stationary series like FRED-MD

    ranks = rng.uniform(-0.5, 0.5, size=(Ttot, N, C)).astype(np.float32)
    # planted SDF weights: linear in char 0,1 + interaction of size (LME, idx 25) and momentum (r12_2, idx 1)
    z1, z2 = ranks[:, :, 0], ranks[:, :, 1]
    w_true = 0.6 * z1 - 0.4 * z2 + 1.5 * (ranks[:, :, 1] * (ranks[:, :, 25] < 0))  # momentum works in small caps
    lam = 0.02 * (1 + 0.8 * np.tanh(state))[:, None]        # time-varying price of risk
    F = lam.squeeze() * 1.0 + 0.04 * rng.standard_normal(Ttot)  # factor return
    beta = w_true / (np.abs(w_true).mean() + 1e-8) * 0.5
    R = beta * F[:, None] + 0.10 * rng.standard_normal((Ttot, N))
    mask = rng.uniform(size=(Ttot, N)) > missing
    R = np.where(mask, R, UNK).astype(np.float32)

    starts = np.cumsum([0] + list(T))
    dates = (1967 + np.arange(Ttot) // 12) * 100 + (np.arange(Ttot) % 12 + 1)
    for name, a, b in zip(['train', 'valid', 'test'], starts[:-1], starts[1:]):
        data = np.concatenate([R[a:b, :, None], ranks[a:b]], axis=2)
        np.savez(os.path.join(out_dir, 'char', f'Char_{name}.npz'), data=data, date=dates[a:b],
                 variable=np.array(['RET'] + names))
        np.savez(os.path.join(out_dir, 'macro', f'macro_{name}.npz'), data=macro[a:b].astype(np.float32),
                 variable=np.array([f'macro_{k}' for k in range(K)]))
    return out_dir
