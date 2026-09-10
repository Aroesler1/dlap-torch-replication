#!/usr/bin/env python
"""
Replicate Chen–Pelger–Zhu (Management Science 2024) "Deep Learning in Asset Pricing" in PyTorch.

    python run.py --config configs/gan.json --logdir output/gan --seeds 9
    python run.py --config configs/gan.json --logdir output/gan --eval_only          # re-evaluate saved seeds
    python run.py --synthetic                                                         # smoke test, no data needed

Outputs in <logdir>/: seed_k/checkpoint.pt, results.json, sdf_factor.csv, weights_*.npy, plots/*.png
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dlap.data import load_all, make_synthetic
from dlap.train import make_bundles, train_one
from dlap import evaluate as ev

DEFAULTS = {  # == authors' config/config.json
    "learning_rate": 0.001, "optimizer": "Adam", "dropout": 0.95, "weighted_loss": True,
    "use_rnn": True, "cell_type_rnn": "lstm", "num_layers_rnn": 1, "num_units_rnn": [4],
    "num_layers": 2, "hidden_dim": [64, 64],
    "cell_type_rnn_moment": "lstm", "num_layers_rnn_moment": 1, "num_units_rnn_moment": [32],
    "num_layers_moment": 0, "hidden_dim_moment": [], "num_condition_moment": 8,
    "num_epochs_unc": 256, "num_epochs_moment": 64, "num_epochs": 1024, "sub_epoch": 4,
    "individual_feature_dim": 46, "macro_feature_dim": 178, "ignore_epoch": 32, "print_freq": 64,   # ignoreEpoch=32: authors' notebook command
    # extensions (0 = off = pure replication)
    "turnover_penalty": 0.0, "sdf_hinge": 0.0,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--logdir', default='output/run')
    ap.add_argument('--seeds', type=int, default=9)
    ap.add_argument('--seed_start', type=int, default=0)
    ap.add_argument('--eval_only', action='store_true')
    ap.add_argument('--synthetic', action='store_true', help='generate fake data in authors\' format and run a tiny config')
    ap.add_argument('--cost_bps', type=float, nargs='*', default=[0, 10, 25, 50], help='one-way costs for net Sharpe')
    ap.add_argument('--beta_epochs', type=int, default=2048)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--set', nargs='*', default=[], help='override config keys, e.g. --set num_epochs=100 turnover_penalty=0.01')
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.synthetic:
        d = make_synthetic('datasets_synthetic', T=(60, 24, 48), N=400, K=20)
        cfg.update({"individual_feature_file": f"{d}/char/Char_train.npz", "individual_feature_file_valid": f"{d}/char/Char_valid.npz",
                    "individual_feature_file_test": f"{d}/char/Char_test.npz", "macro_feature_file": f"{d}/macro/macro_train.npz",
                    "macro_feature_file_valid": f"{d}/macro/macro_valid.npz", "macro_feature_file_test": f"{d}/macro/macro_test.npz",
                    "macro_feature_dim": 20, "num_epochs_unc": 40, "num_epochs_moment": 20, "num_epochs": 60,
                    "ignore_epoch": 5, "print_freq": 20})
        args.seeds = min(args.seeds, 2); args.beta_epochs = min(args.beta_epochs, 200)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    for kv in args.set:
        k, v = kv.split('=', 1); cfg[k] = json.loads(v) if v[0] in '[{"0123456789-.tf' else v
    os.makedirs(args.logdir, exist_ok=True)
    logf = open(os.path.join(args.logdir, 'log.txt'), 'a')

    def log(s):
        print(s); logf.write(s + '\n'); logf.flush()

    log('=' * 90); log(time.strftime('%Y-%m-%d %H:%M:%S') + '  config: ' + json.dumps(cfg))

    tr, va, te = load_all(cfg)
    cfg['individual_feature_dim'] = tr.C; cfg['macro_feature_dim'] = tr.K
    for s, n in zip((tr, va, te), ('train', 'valid', 'test')):
        log(s.summary(n))
    bundles = make_bundles(tr, va, te, args.device)

    # ---------------- training ----------------
    ckpts = []
    for k in range(args.seed_start, args.seed_start + args.seeds):
        d = os.path.join(args.logdir, f'seed_{k}'); p = os.path.join(d, 'checkpoint.pt')
        if not args.eval_only or not os.path.exists(p):
            log(f'\n##### seed {k} #####'); t0 = time.time()
            train_one(cfg, bundles, d, seed=k, log=log, device=args.device)
            log(f'seed {k} done in {time.time() - t0:.0f}s')
        ckpts.append(p)

    # ---------------- ensemble evaluation ----------------
    log('\n##### ensemble evaluation #####')
    W = ev.ensemble_weights(cfg, ckpts, bundles, device=args.device)
    splits = (tr, va, te); names = ('train', 'valid', 'test')
    res = {'config': cfg, 'n_seeds': len(ckpts), 'paper_reference_test': {
        'SR_monthly': 0.75, 'SR_annual': 2.6, 'EV': 0.08, 'XS_R2': 0.23,
        'note': 'GAN with hidden macro states, Table 1 of the paper; the paper XS_R2 is the T_i-weighted one (notebook prints WXSR2) -> compare with XS_R2_weighted_beta'}}
    Wn, Fs = [], []
    for s, n, w in zip(splits, names, W):
        wn = ev.normalize_w(w, s.mask); F = ev.sdf_factor_from_w(wn, s.R, s.mask)
        Wn.append(wn); Fs.append(F)
        to = ev.turnover(wn, s.mask)
        e, xs, xsw = ev.ev_xsr2(w, s.R, s.mask)
        r = {'SR_monthly': ev.sharpe_monthly(F), 'SR_annual': ev.sharpe_monthly(F) * np.sqrt(12),
             'mean_monthly_return_%': float(F.mean() * 100), 'vol_monthly_%': float(F.std() * 100),
             'max_drawdown': ev.max_drawdown(F), 'worst_month': float(F.min()),
             'EV_wproj': e, 'XS_R2_wproj': xs, 'XS_R2_weighted_wproj': xsw,
             'turnover_mean': float(to.mean()),
             'SR_annual_net_of_cost': {f'{c:g}bps': ev.sharpe_monthly(ev.net_of_cost_factor(F, to, c)) * np.sqrt(12) for c in args.cost_bps},
             'SR_annual_by_year': ev.yearly_sharpe(F, s.dates)}
        res[n] = r
        log(f"{n:6s} SR monthly {r['SR_monthly']:.3f}  annual {r['SR_annual']:.2f}  EV {e:.3f}  XS-R2 {xs:.3f}  "
            f"turnover {to.mean():.2f}  net SR@{args.cost_bps[-1]:g}bps {list(r['SR_annual_net_of_cost'].values())[-1]:.2f}")
        np.save(os.path.join(args.logdir, f'weights_{n}.npy'), wn)

    # ---------------- beta network (EV / XS-R2 as reported in the paper) ----------------
    log('fitting beta network on SDF factor ...')
    betas = ev.fit_beta(splits, Fs, epochs=args.beta_epochs, seeds=tuple(range(min(len(ckpts), 9))), device=args.device, log=log)
    for s, n, b in zip(splits, names, betas):
        e, xs, xsw = ev.ev_xsr2(b, s.R, s.mask)
        res[n].update({'EV_beta': e, 'XS_R2_beta': xs, 'XS_R2_weighted_beta': xsw})
        dec = ev.decile_portfolios(b, s.R, s.mask)
        res[n]['beta_decile_mean_return_%'] = [float(x) for x in np.nanmean(dec, 0) * 100]
        log(f"{n:6s} beta-net  EV {e:.3f}  XS-R2 {xs:.3f}  weighted XS-R2 {xsw:.3f}  decile spread {(np.nanmean(dec,0)[-1]-np.nanmean(dec,0)[0])*100:.2f}%/mo")

    # ---------------- save ----------------
    import csv
    with open(os.path.join(args.logdir, 'sdf_factor.csv'), 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['date', 'split', 'F'])
        for s, n, F in zip(splits, names, Fs):
            for d, x in zip(s.dates, F): wr.writerow([int(d), n, float(x)])
    with open(os.path.join(args.logdir, 'results.json'), 'w') as f:
        json.dump(res, f, indent=2, default=float)
    try:
        make_plots(args.logdir, cfg, ckpts, splits, Fs, betas, bundles, args.device)
    except Exception as e:  # plots are optional
        log(f'plotting failed: {e}')
    log(f'\nresults written to {args.logdir}/results.json')


def make_plots(logdir, cfg, ckpts, splits, Fs, betas, bundles, device):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    pd = os.path.join(logdir, 'plots'); os.makedirs(pd, exist_ok=True)
    tr, va, te = splits
    # cumulative SDF factor
    F = np.concatenate(Fs); dates = np.concatenate([s.dates for s in splits])
    x = np.arange(len(F))
    plt.figure(figsize=(10, 4)); plt.plot(x, np.cumsum(F)); 
    for cut in (tr.T, tr.T + va.T): plt.axvline(cut, color='k', ls='--', lw=.8)
    tick = np.arange(0, len(F), 60); plt.xticks(tick, [str(int(d) // 100) for d in dates[tick]])
    plt.title('Cumulative return of the SDF factor (L1-normalised weights) — train | valid | test'); plt.tight_layout()
    plt.savefig(os.path.join(pd, 'cumulative_sdf_factor.png'), dpi=150); plt.close()
    # training curves
    hist = torch.load(ckpts[0], map_location='cpu', weights_only=False)['hist']
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k in ('sr_tr', 'sr_va', 'sr_te'): ax[0].plot(hist[k], label=k)
    ax[0].axvline(cfg['num_epochs_unc'], color='k', ls='--', lw=.8); ax[0].legend(); ax[0].set_title('monthly Sharpe by epoch (seed 0)')
    for k in ('loss_tr', 'loss_va', 'loss_te'): ax[1].semilogy(hist[k], label=k)
    ax[1].axvline(cfg['num_epochs_unc'], color='k', ls='--', lw=.8); ax[1].legend(); ax[1].set_title('unconditional pricing loss')
    plt.tight_layout(); plt.savefig(os.path.join(pd, 'training_curves.png'), dpi=150); plt.close()
    # macro hidden states (seed 0)
    st = ev.macro_states(cfg, ckpts[0], bundles[2], device)
    if st is not None:
        plt.figure(figsize=(10, 3.5)); plt.plot(st); plt.title('LSTM macro hidden states, test window (seed 0)')
        tick = np.arange(0, te.T, 36); plt.xticks(tick, [str(int(d) // 100) for d in te.dates[tick]])
        plt.tight_layout(); plt.savefig(os.path.join(pd, 'macro_states_test.png'), dpi=150); plt.close()
    # beta deciles (test)
    dec = np.nanmean(ev.decile_portfolios(betas[2], te.R, te.mask), 0) * 100
    plt.figure(figsize=(6, 3.5)); plt.bar(np.arange(1, 11), dec); plt.xlabel('beta decile'); plt.ylabel('mean monthly excess return (%)')
    plt.title('Test: returns of SDF-beta sorted deciles'); plt.tight_layout(); plt.savefig(os.path.join(pd, 'beta_deciles_test.png'), dpi=150); plt.close()
    # yearly Sharpe (test)
    ys = ev.yearly_sharpe(Fs[2], te.dates)
    plt.figure(figsize=(10, 3.5)); plt.bar(list(ys.keys()), list(ys.values())); plt.title('Test: annualised Sharpe by calendar year')
    plt.tight_layout(); plt.savefig(os.path.join(pd, 'sharpe_by_year_test.png'), dpi=150); plt.close()
    # variable importance: authors' `_saveIndividualFeatureImportance` (finite difference, delta 1e-6, mean |dw| on the
    # test window, raw w) averaged over the ensemble members as in `plotIndividualFeatureImportance`
    from dlap.model import SDFNet
    b = bundles[2]; delta = 1e-6; imp = np.zeros(tr.C)
    for p in ckpts:
        ck = torch.load(p, map_location=device, weights_only=False)
        net = SDFNet(cfg).to(device); net.load_state_dict(ck['sdf_best_sharpe'] or ck['sdf_last']); net.eval()
        with torch.no_grad():
            state = net.macro_state(b.macro_seq); w0 = net.weights(b.I, b.mask, state, b.t0)
            for j in range(tr.C):
                I2 = b.I.clone(); I2[..., j] += delta
                imp[j] += ((net.weights(I2, b.mask, state, b.t0) - w0).abs().mean() / delta).item() / len(ckpts)
    imp = list(imp)
    order = np.argsort(imp)[::-1]
    plt.figure(figsize=(10, 4)); plt.bar(range(tr.C), np.array(imp)[order]); plt.xticks(range(tr.C), [tr.char_names[i] for i in order], rotation=90, fontsize=7)
    plt.title(f'Variable importance: avg |dw/dchar| on test ({len(ckpts)}-model average)'); plt.tight_layout(); plt.savefig(os.path.join(pd, 'variable_importance_test.png'), dpi=150); plt.close()
    json.dump({tr.char_names[i]: imp[i] for i in order}, open(os.path.join(logdir, 'variable_importance.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
