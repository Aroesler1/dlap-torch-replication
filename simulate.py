#!/usr/bin/env python
"""
Replicate Table II of Chen–Pelger–Zhu ("Simulation Example") — no CRSP data needed.

DGP (Section IV of the paper):   R^e_{t+1,i} = beta_{t,i} F_{t+1} + eps_{t+1,i},
    F ~ N(mu_F, sigma_F^2) i.i.d. with sigma_F^2 = 0.1 and Sharpe 1  (=> mu_F = sqrt(0.1)),  eps ~ N(0, 1)
    Setup 1 "two characteristics":       beta = C1 * C2,           C ~ N(0,1)
    Setup 2 "one char + macro state":    beta = C * b(h_t),        h_t = sin(pi t/24) + N(0, 0.25),  b(h) = sign(h)
                                          observed macro series   Z_t = 0.05 t + h_t   (network sees the increment dZ_t)
    N = 500, T = 600 = 250 train / 100 valid / 250 test.

    python simulate.py --setup 1 --seeds 3
    python simulate.py --setup 2 --seeds 3
"""
import os, sys, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dlap.data import Split, UNK
from dlap.train import make_bundles, train_one
from dlap import evaluate as ev
from run import DEFAULTS

PAPER = {1: {'Population': (0.96, 1.09, 0.94, 0.16, 0.15, 0.17, 0.17, 0.15, 0.17),
             'GAN': (0.98, 1.11, 0.94, 0.12, 0.11, 0.13, 0.10, 0.09, 0.07),
             'FFN': (0.94, 1.04, 0.89, 0.05, 0.04, 0.05, -0.30, -0.09, -0.33),
             'LS': (0.07, -0.10, 0.01, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01)},
         2: {'Population': (0.89, 0.92, 0.86, 0.18, 0.18, 0.17, 0.19, 0.20, 0.15),
             'GAN': (0.79, 0.77, 0.64, 0.18, 0.18, 0.17, 0.19, 0.20, 0.15),
             'FFN': (0.05, -0.05, 0.06, 0.02, 0.01, 0.02, 0.01, 0.01, 0.02),
             'LS': (0.12, -0.05, 0.10, 0.16, 0.16, 0.15, 0.15, 0.18, 0.14)}}


def generate(setup, out, seed=0, N=500, T=(250, 100, 250)):
    rng = np.random.default_rng(seed)
    Ttot = sum(T); sF = np.sqrt(0.1); muF = sF * 1.0
    F = muF + sF * rng.standard_normal(Ttot)
    if setup == 1:
        C = rng.standard_normal((Ttot, N, 2)); beta = C[:, :, 0] * C[:, :, 1]
        macro = np.zeros((Ttot, 0)); h = None
    else:
        C = rng.standard_normal((Ttot, N, 1))
        t = np.arange(1, Ttot + 1)
        h = np.sin(np.pi * t / 24) + rng.normal(0, np.sqrt(0.25), Ttot)
        Z = 0.05 * t + h
        dZ = np.diff(Z, prepend=Z[0])
        macro = dZ[:, None]                                      # increment only: LSTM has to recover the cycle
        beta = C[:, :, 0] * np.where(h > 0, 1.0, -1.0)[:, None]
    R = beta * F[:, None] + rng.standard_normal((Ttot, N))
    os.makedirs(f'{out}/char', exist_ok=True); os.makedirs(f'{out}/macro', exist_ok=True)
    dates = 190001 + (np.arange(Ttot) // 12) * 100 + np.arange(Ttot) % 12
    a = np.cumsum([0] + list(T))
    for name, s, e in zip(['train', 'valid', 'test'], a[:-1], a[1:]):
        np.savez(f'{out}/char/Char_{name}.npz', data=np.concatenate([R[s:e, :, None], C[s:e]], 2).astype(np.float32),
                 date=dates[s:e], variable=np.array(['RET'] + [f'C{k+1}' for k in range(C.shape[2])]))
        np.savez(f'{out}/macro/macro_{name}.npz', data=macro[s:e].astype(np.float32), variable=np.array(['dZ']))
    return F, beta, h, a


def metrics(w_dense_list, beta_list, splits, sign=1.0):
    """(SR train/valid/test, EV train/valid/test, XS train/valid/test) from weights and loadings.
    sdf_factor_from_w already returns F = -w'R (authors' convention for the network output);
    pass sign=-1 for models whose weights are +omega (population beta, FFN mu, LS theta)."""
    out = []
    for w, b, s in zip(w_dense_list, beta_list, splits):
        Fh = sign * ev.sdf_factor_from_w(ev.normalize_w(w, s.mask), s.R, s.mask)
        out.append(ev.sharpe_monthly(Fh))
    for b, s in zip(beta_list, splits):
        out.append(ev.ev_xsr2(b, s.R, s.mask)[0])
    for b, s in zip(beta_list, splits):
        out.append(ev.ev_xsr2(b, s.R, s.mask)[1])
    return out


def ls_model(splits):
    """Linear special case: theta = (1/T sum F~F~')^{-1}(1/T sum F~),  F~_t = 1/N sum_i I_{t,i} R_{t+1,i}; beta by OLS of R F on I."""
    tr = splits[0]
    Ft = np.stack([(tr.I[t][tr.mask[t]] * tr.R[t][tr.mask[t], None]).mean(0) for t in range(tr.T)])   # [T, C]
    theta = np.linalg.solve(Ft.T @ Ft / tr.T + 1e-8 * np.eye(Ft.shape[1]), Ft.mean(0))
    W = [s.I @ theta for s in splits]
    F = [ev.sdf_factor_from_w(ev.normalize_w(w, s.mask), s.R, s.mask) for w, s in zip(W, splits)]
    X = tr.I[tr.mask]; y = (tr.R * F[0][:, None])[tr.mask]
    Xb = np.c_[X, np.ones(len(X))]; coef = np.linalg.lstsq(Xb, y, rcond=None)[0]
    B = [np.c_[s.I.reshape(-1, s.C), np.ones(s.T * s.N)] @ coef for s in splits]
    B = [b.reshape(s.T, s.N) for b, s in zip(B, splits)]
    return W, B


def ffn_model(splits, device, epochs=2048, seeds=(0, 1, 2)):
    """Forecasting approach (Gu–Kelly–Xiu FFN [32,16,8]): mu = E[R|I]; used as both omega and beta."""
    mu = ev.fit_beta(splits, [np.ones(s.T) for s in splits], epochs=epochs, seeds=seeds, device=device,
                     log=lambda *_: None, scale=1.0)
    return mu, mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setup', type=int, default=1)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--data_seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--logdir', default=None)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--beta_epochs', type=int, default=2048)
    args = ap.parse_args()
    logdir = args.logdir or f'output/sim{args.setup}'
    os.makedirs(logdir, exist_ok=True)
    d = f'datasets_sim{args.setup}'
    F_true, beta_true, h_true, cuts = generate(args.setup, d, args.data_seed)

    cfg = dict(DEFAULTS)
    cfg.update({"individual_feature_file": f"{d}/char/Char_train.npz", "individual_feature_file_valid": f"{d}/char/Char_valid.npz",
                "individual_feature_file_test": f"{d}/char/Char_test.npz", "macro_feature_file": f"{d}/macro/macro_train.npz",
                "macro_feature_file_valid": f"{d}/macro/macro_valid.npz", "macro_feature_file_test": f"{d}/macro/macro_test.npz",
                "macro_feature_dim": 0 if args.setup == 1 else 1, "individual_feature_dim": 2 if args.setup == 1 else 1,
                "print_freq": 128})
    for kv in args.set:
        k, v = kv.split('=', 1); cfg[k] = json.loads(v)
    from dlap.data import load_all
    splits = load_all(cfg); tr, va, te = splits
    bundles = make_bundles(tr, va, te, args.device)

    rows = {}
    # population
    Bp = [beta_true[s:e] for s, e in zip(cuts[:-1], cuts[1:])]
    rows['Population'] = metrics(Bp, Bp, splits, sign=-1)  # F = +beta'R (sdf_factor_from_w returns -w'R)
    # GAN
    ck = []
    for k in range(args.seeds):
        t0 = time.time()
        train_one(cfg, bundles, f'{logdir}/seed_{k}', seed=k, log=print, device=args.device)
        print(f'seed {k}: {time.time()-t0:.0f}s'); ck.append(f'{logdir}/seed_{k}/checkpoint.pt')
    W = ev.ensemble_weights(cfg, ck, bundles, device=args.device)
    Fg = [ev.sdf_factor_from_w(ev.normalize_w(w, s.mask), s.R, s.mask) for w, s in zip(W, splits)]
    Bg = ev.fit_beta(splits, Fg, epochs=args.beta_epochs, seeds=tuple(range(args.seeds)), device=args.device, log=lambda *_: None)
    rows['GAN'] = metrics(W, Bg, splits)                  # net w is minus omega; F = 1 - M = -w'R already
    rows['GAN (w as loading)'] = metrics(W, W, splits)
    # FFN, LS
    Wf, Bf = ffn_model(splits, args.device, epochs=args.beta_epochs); rows['FFN'] = metrics(Wf, Bf, splits, sign=-1)
    Wl, Bl = ls_model(splits); rows['LS'] = metrics(Wl, Bl, splits, sign=-1)

    hdr = ['SR tr', 'SR va', 'SR te', 'EV tr', 'EV va', 'EV te', 'XS tr', 'XS va', 'XS te']
    print(f'\n=== Table II replication, setup {args.setup} ({args.seeds} GAN seeds, data seed {args.data_seed}) ===')
    print(f"{'model':22s}" + ''.join(f'{h:>8s}' for h in hdr))
    for m, r in rows.items():
        print(f'{m:22s}' + ''.join(f'{x:8.2f}' for x in r))
        if m in PAPER[args.setup]:
            print(f'{"  paper":22s}' + ''.join(f'{x:8.2f}' for x in PAPER[args.setup][m]))
    json.dump({'ours': rows, 'paper': PAPER[args.setup], 'cfg': cfg}, open(f'{logdir}/table2.json', 'w'), indent=1, default=float)

    # hidden state recovery plot (Figure 5 analogue) for setup 2
    if args.setup == 2:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        st = np.concatenate([ev.macro_states(cfg, ck[0], b, args.device) for b in bundles])
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax[0].plot(h_true, lw=.8); ax[0].axhline(0, color='k', lw=.5); ax[0].set_title('true hidden state h_t (unobserved)')
        ax[1].plot(st, lw=.8); ax[1].set_title('LSTM hidden states recovered from the increment dZ_t (seed 0)')
        for a in ax:
            for c in cuts[1:-1]: a.axvline(c, color='k', ls='--', lw=.8)
        plt.tight_layout(); plt.savefig(f'{logdir}/hidden_state_recovery.png', dpi=150)
        corr = [abs(np.corrcoef(st[:, j], h_true)[0, 1]) for j in range(st.shape[1])]
        print(f'|corr(LSTM state, true h)| by unit: {np.round(corr, 2)}')


if __name__ == '__main__':
    main()
