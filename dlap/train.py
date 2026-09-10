"""
Three-stage adversarial training, exactly as in the authors' `train()`:

  Stage 1  (num_epochs_unc, default 256): SDF net on the *unconditional* loss (g = 1).
           keep checkpoint with best validation loss ("loss") and best validation Sharpe ("sharpe").
  Stage 2  (num_epochs_moment, default 64): freeze SDF net (from best-loss ckpt), train the adversary
           to MAXIMISE the conditional loss; keep the adversary that achieved the largest loss.
  Stage 3  (num_epochs, default 1024): freeze adversary, retrain SDF net on the conditional loss;
           again keep best-valid-loss and best-valid-Sharpe checkpoints.

Each "epoch" = `sub_epoch` (default 4) full-batch Adam steps on the training window, then evaluation.
The authors' notebook ensembles the *best-valid-Sharpe* checkpoint of each of 9 seeds.
"""
import os, time, json, copy
import numpy as np
import torch

from .model import SDFNet, MomentNet, pricing_loss, turnover_penalty, sdf_hinge
from .evaluate import sharpe_monthly


class Bundle:
    """Tensors for one split + the macro history needed to run the LSTM from t=0 of the training window."""

    def __init__(self, split, macro_prefix, device):
        self.I = torch.tensor(split.I, device=device)
        self.R = torch.tensor(split.R, device=device)
        self.mask = torch.tensor(split.mask, device=device)
        self.T_i = torch.tensor(split.T_i(), device=device, dtype=torch.float32)
        self.macro_seq = torch.tensor(np.concatenate(macro_prefix + [split.M], 0), device=device)
        self.t0 = int(sum(m.shape[0] for m in macro_prefix))
        self.T = split.T


def make_bundles(tr, va, te, device):
    return (Bundle(tr, [], device),
            Bundle(va, [tr.M], device),
            Bundle(te, [tr.M, va.M], device))


@torch.no_grad()
def evaluate(sdf, b, weighted=True):
    sdf.eval()
    state = sdf.macro_state(b.macro_seq)
    w, W, M = sdf(b.I, b.R, b.mask, state, b.t0)
    loss = pricing_loss(M, b.R, b.mask, None, b.T_i, weighted).item()
    sr = sharpe_monthly((1.0 - M).cpu().numpy())
    return loss, sr


def _step(opt, loss):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def train_one(cfg, bundles, logdir, seed=0, log=print, device='cuda'):
    torch.manual_seed(seed); np.random.seed(seed)
    btr, bva, bte = bundles
    os.makedirs(logdir, exist_ok=True)
    weighted = cfg.get('weighted_loss', True)
    lam_to = cfg.get('turnover_penalty', 0.0)
    lam_pos = cfg.get('sdf_hinge', 0.0)
    sub = cfg.get('sub_epoch', 4) or 1
    ignore = cfg.get('ignore_epoch', 64)

    sdf = SDFNet(cfg).to(device)
    adv = MomentNet(cfg).to(device)
    opt_sdf = torch.optim.Adam(sdf.parameters(), lr=cfg['learning_rate'])
    opt_adv = torch.optim.Adam(adv.parameters(), lr=cfg['learning_rate'])

    hist = {'stage': [], 'epoch': [], 'loss_tr': [], 'loss_va': [], 'loss_te': [], 'sr_tr': [], 'sr_va': [], 'sr_te': []}
    best = {'loss': (float('inf'), None), 'sharpe': (float('-inf'), None)}

    def sdf_loss(G):
        sdf.train()
        state = sdf.macro_state(btr.macro_seq)
        w, W, M = sdf(btr.I, btr.R, btr.mask, state, btr.t0)
        loss = pricing_loss(M, btr.R, btr.mask, G, btr.T_i, weighted)
        if lam_to > 0:
            loss = loss + lam_to * turnover_penalty(W, btr.mask)
        if lam_pos > 0:
            loss = loss + lam_pos * sdf_hinge(M)
        return loss

    def record(stage, epoch, t_start, n_epochs):
        ltr, srtr = evaluate(sdf, btr, weighted)
        lva, srva = evaluate(sdf, bva, weighted)
        lte, srte = evaluate(sdf, bte, weighted)
        for k, v in zip(['stage', 'epoch', 'loss_tr', 'loss_va', 'loss_te', 'sr_tr', 'sr_va', 'sr_te'],
                        [stage, epoch, ltr, lva, lte, srtr, srva, srte]):
            hist[k].append(v)
        if epoch > ignore:
            if lva < best['loss'][0]:
                best['loss'] = (lva, copy.deepcopy(sdf.state_dict()))
            if srva > best['sharpe'][0]:
                best['sharpe'] = (srva, copy.deepcopy(sdf.state_dict()))
        if epoch % cfg.get('print_freq', 64) == 0 or epoch == n_epochs - 1:
            el = time.time() - t_start
            log(f"[{stage}] ep {epoch:4d}/{n_epochs}  loss tr/va/te {ltr:.3e}/{lva:.3e}/{lte:.3e}  "
                f"SR(monthly) tr/va/te {srtr:.3f}/{srva:.3f}/{srte:.3f}   {el:6.0f}s")

    # ---------------- Stage 1: unconditional ----------------
    t0 = time.time(); log('Stage 1: unconditional loss')
    for ep in range(cfg['num_epochs_unc']):
        for _ in range(sub):
            _step(opt_sdf, sdf_loss(None))
        record('UNC', ep, t0, cfg['num_epochs_unc'])

    # ---------------- Stage 2: adversary ----------------
    log('Stage 2: updating moment conditions (adversary)')
    sdf.load_state_dict(best['loss'][1] if best['loss'][1] is not None else sdf.state_dict())
    sdf.eval()
    with torch.no_grad():
        state = sdf.macro_state(btr.macro_seq)
        _, _, M_fixed = sdf(btr.I, btr.R, btr.mask, state, btr.t0)
    best_adv, best_adv_loss = copy.deepcopy(adv.state_dict()), float('-inf')
    for ep in range(cfg['num_epochs_moment']):
        adv.train()
        G = adv(btr.I, btr.mask, btr.macro_seq, btr.t0)
        loss = pricing_loss(M_fixed, btr.R, btr.mask, G, btr.T_i, weighted)
        _step(opt_adv, -loss)
        if loss.item() > best_adv_loss:
            best_adv_loss, best_adv = loss.item(), copy.deepcopy(adv.state_dict())
    adv.load_state_dict(best_adv); adv.eval()
    with torch.no_grad():
        G_fixed = adv(btr.I, btr.mask, btr.macro_seq, btr.t0).detach()
    log(f'   adversary conditional loss {best_adv_loss:.3e}  (unconditional was {best["loss"][0]:.3e})')

    # ---------------- Stage 3: conditional ----------------
    if cfg['num_epochs'] > 0:   # fresh selection for stage 3, as in authors' code (UNC ablation keeps stage-1 best)
        best = {'loss': (float('inf'), None), 'sharpe': (float('-inf'), None)}
    t0 = time.time(); log('Stage 3: conditional (GAN) loss')
    for ep in range(cfg['num_epochs']):
        for _ in range(sub):
            _step(opt_sdf, sdf_loss(G_fixed))
        record('GAN', ep, t0, cfg['num_epochs'])

    torch.save({'cfg': cfg, 'seed': seed,
                'sdf_best_sharpe': best['sharpe'][1], 'sdf_best_loss': best['loss'][1],
                'sdf_last': sdf.state_dict(), 'adv': adv.state_dict(), 'hist': hist},
               os.path.join(logdir, 'checkpoint.pt'))
    with open(os.path.join(logdir, 'history.json'), 'w') as f:
        json.dump(hist, f)
    off = cfg['num_epochs_unc'] if cfg['num_epochs'] > 0 else 0
    i = int(np.argmax(hist['sr_va'][off:])) + off
    log(f"best-valid-Sharpe epoch {hist['epoch'][i]}: SR(monthly) tr/va/te = "
        f"{hist['sr_tr'][i]:.3f}/{hist['sr_va'][i]:.3f}/{hist['sr_te'][i]:.3f}")
    return hist
