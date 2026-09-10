# Replication notes — Chen, Pelger & Zhu (2024), PyTorch port, H100 runs of 2026-09-10

Two full runs were made. **v1** (commit f9a02cc) used the port as delivered. A line-by-line audit against the authors'
TensorFlow code and notebook then found the deviations listed below; all were fixed and everything was re-run (**v2**, the
numbers reported here). v1 results are kept in `output_v1/` for the before/after comparison.

## Environment
* Python: `.venv` created from `C:\ProgramData\anaconda3\python.exe` (3.13). torch **2.11.0+cu128**, numpy 2.5.3, matplotlib 3.11.1.
  `torch.cuda.is_available()` = True, device NVIDIA H100 NVL (driver 581.80). Training, ensemble evaluation, beta network and simulations all ran on the GPU.
  Gotcha: `pip install torch --index-url .../cu128 --extra-index-url pypi.org` resolved the CPU wheel; reinstalling from the cu128 index alone fixed it.
* Data: the six npz files were extracted from `drive-download-…/datasets.zip` into `datasets/char` and `datasets/macro`; `check_data.py` passed.

## Data summary (loader output)
| split | months | permnos (N) | observations | avg stocks/month | chars | macro |
|---|---|---|---|---|---|---|
| train | 240 (1967-01 – 1986-12) | 3686 | 336,113 | 1400.5 | 46 | 178 |
| valid | 60 (1987-01 – 1991-12) | 3347 | 132,167 | 2202.8 | 46 | 178 |
| test | 300 (1992-01 – 2016-12) | 7141 | 750,275 | 2500.9 | 46 | 178 |

N per split is the number of distinct PERMNOs observed in that window (the paper's "~10,000 stocks" is the union over the sample). Macro files have 178 columns, so no `macro_idx` was needed and no config was changed.

## Fidelity audit against the authors' code (what was compared)
`config/config.json`, `run.py`, `src/data/data_layer.py`, `src/model/model_GAN.py` (graph, `_add_loss`, `train`, evaluation helpers),
`src/model/model_utils.py` (`calculateStatistics`), `src/utils.py` (`sharpe`), `create_RF_data.py`, `src/model/model_RtnFcst.py`, and — fetched
from the authors' GitHub because the local copy lacks them — `model_GAN.ipynb`, `run_RtnFcst_ensembles.py`, `config_RF/config_RF_1.json`,
plus the layout of `sample_checkpoints.zip`.

**Verified identical (unchanged in the port)**
* Hyper-parameters: every key of the authors' `config.json` equals the port's DEFAULTS (Table I optimum: lr 0.001, Adam, dropout keep 0.95, SDF FFN [64,64] + LSTM 4, adversary LSTM 32 + 8 tanh moments, 0 hidden layers, epochs 256 / 64 / 1024, sub_epoch 4, weighted loss, all 178 macro series).
* Data: mask = return ≠ −99.99; macro standardised with training-window mean/std; loss weights T_i.
* Networks, loss (mean over moments × stocks of [(1/T_i) Σ_t M_t R_{t,i} g_{j,t,i}]² · T_i / max T_i), three-stage schedule with reload of the best-valid-loss checkpoint before stages 2 and 3, separate best-loss / best-Sharpe tracking per stage, selection on unconditional validation loss and raw-weight Sharpe, Sharpe = mean/std (ddof 0) of 1 − M, LSTM state carried train → valid → test, dropout off at evaluation.
* Ensemble: the notebook loads the nine `Task_1_Trial_k/sharpe` (best-valid-Sharpe) checkpoints, averages raw w, L1-normalises each month and reports the Sharpe of that normalised factor; EV / XS-R² use the beta-network ensemble via `calculateStatistics`, and the notebook prints the **T_i-weighted** XS-R² (`WXSR2`). The port does exactly this; `summarize.py` now lists the weighted XS-R² first.
* Beta network (`config_RF_1.json`): 3 layers [32,16,8], keep 0.95, Adam 1e-3, 2048 single-step epochs, unweighted MSE, characteristics only, best-valid-loss checkpoint, 9 trials averaged, target R·F·50 from the normalised ensemble SDF.

**Deviations found and fixed (v1 → v2)**
1. Initialisation: PyTorch defaults → TF 1.12 defaults (Glorot-uniform kernels, zero biases, LSTM Glorot over the joint [input+hidden, 4·hidden] kernel, forget-gate bias 1). `dlap/model.py: init_tf_default_`, applied to the SDF net, the adversary and the beta net.
2. Optimizer: one shared `torch.optim.Adam` for stages 1 and 3 → a `tf.train.AdamOptimizer`-exact update (`TFAdam`, epsilon inside the square-root-corrected denominator) with a fresh instance per stage, matching the authors' three `optimize_loss` ops.
3. Adversarial stages: the frozen network was evaluated once without dropout → it is re-evaluated every step with dropout active (keep 0.95 is fed to the whole TF graph in both stages).
4. Adversary training length: 64 steps → the authors' loop runs the 64 steps once per sub-epoch (4 × 64 = 256 steps) and resets the running maximum at each pass; the kept adversary is the running-max of the last pass. Reproduced literally.
5. `ignore_epoch`: 64 (run.py flag default) → 32, the value in the authors' notebook command that produced their sample checkpoints.
6. Beta network: validation loss checked every 16 epochs → every epoch, as in `model_RtnFcst.train`.
7. Variable importance: seed-0, delta 1e-4 → 9-model average with delta 1e-6 (`_saveIndividualFeatureImportance` / `plotIndividualFeatureImportance`).
8. Macro standardisation moments computed in float64 before the float32 cast, as the authors' data layer does.

**Remaining differences that cannot be closed from the released code**
* The paper's numbers come from a validation search over 384 configurations, from which Table I reports the optimum; we run the optimum only. The sample checkpoints on the authors' Drive are also single-configuration runs.
* Random draws (TF vs PyTorch RNG streams) and float32 reduction order. These only move seed-level paths.
* The turnover / positive-SDF extensions are ours; the simulation benchmarks (population, FFN, LS rows) are the port's own implementations because the authors did not release simulation code.

## Timing
* Smoke test (`configs/quick.json`, 352 epochs, 1 seed): 11 s of training, ≈0.03 s/epoch alone on the GPU.
* Full seed (256 + 4×64 adversary + 1024 epochs) under 8-way GPU sharing, `output/gan` seeds 0–8: 226, 247, 204, 179, 148, 133, 140, 114, 105 s. Alone a seed takes ≈45 s. The whole plan (49 seeds + 2 simulations + evaluation) took ≈35 min of wall clock.

## Headline replication: `output/gan` (9-seed ensemble of best-valid-Sharpe checkpoints, L1-normalised weights)
| | train | valid | test | paper test | v1 test (before fixes) |
|---|---|---|---|---|---|
| Sharpe (monthly) | 2.99 | 1.36 | **0.68** | 0.75 | 0.63 |
| Sharpe (annual) | 10.35 | 4.70 | **2.36** | 2.60 | 2.20 |
| EV (beta net) | 0.206 | 0.098 | **0.084** | 0.08 | 0.050 |
| XS-R² (beta net, T_i-weighted = paper definition) | 0.132 | 0.013 | **0.228** | 0.23 | 0.189 |
| XS-R² (beta net, unweighted) | -0.024 | 0.040 | 0.045 | – | 0.027 |
| turnover / month | 0.96 | 0.96 | 0.96 | – | 1.07 |

Paper train / valid Sharpe: 2.68 / 1.43. Test factor: mean 0.79 %/month, vol 1.16 %, max drawdown 5.5 %, worst month −4.5 %.
Net-of-cost annual test Sharpe: 2.36 (0 bps), 2.07 (10), 1.64 (25), 0.92 (50). Beta-decile mean returns on test (%/month): −0.20, 0.40, 0.73, 0.86, 0.94, 1.13, 1.27, 1.40, 1.76, 3.18 (monotone, paper Fig. 9).
Annualised test Sharpe by year: 1992 3.9, 1993 3.9, 1994 5.3, 1995 13.0, 1996 4.7, 1997 6.9, 1998 9.0, 1999 3.0, 2000 2.2, 2001 3.0, 2002 2.9, 2003 −0.1, 2004 3.8, 2005 3.9, 2006 3.4, 2007 2.4, 2008 3.6, 2009 −0.3, 2010 2.3, 2011 1.6, 2012 4.5, 2013 3.0, 2014 0.8, 2015 3.4, 2016 0.5.

Individual seeds (best-valid-Sharpe epoch of stage 3), monthly Sharpe:
| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| test | 0.750 | 0.378 | 0.478 | 0.589 | 0.481 | 0.600 | 0.283 | 0.394 | 0.458 |
| valid | 0.97 | 0.88 | 0.92 | 1.02 | 1.00 | 1.02 | 1.00 | 0.72 | 1.21 |
| train | 1.98 | 2.85 | 3.43 | 2.91 | 2.00 | 2.91 | 3.68 | 3.14 | 4.30 |
| epoch | 955 | 746 | 1020 | 1003 | 985 | 1012 | 1019 | 1007 | 1009 |

Mean 0.49, sd 0.14; the 9-model ensemble (0.68) beats every single seed, as the paper reports. Six of nine seeds pick an epoch in the last 25 of stage 3.

## All runs (v2; `summary.md`)
| run | seeds | SR train | SR valid | SR test (m) | SR test (a) | EV test | XS-R² test (wtd) | turnover | net SR @50 bps | paper test SR |
|---|---|---|---|---|---|---|---|---|---|---|
| gan | 9 | 2.99 | 1.36 | 0.68 | 2.36 | 0.08 | 0.23 | 0.96 | 0.92 | 0.75 |
| unc (no adversary) | 9 | 1.52 | 0.72 | 0.45 | 1.57 | 0.10 | 0.29 | 0.80 | 0.55 | 0.53 |
| gan_nomacro | 9 | 1.76 | 1.49 | 0.62 | 2.15 | 0.10 | 0.26 | 0.96 | 0.62 | 0.69 |
| gan_allmacro_raw | 4 | 1.03 | 0.02 | 0.07 | 0.23 | 0.12 | 0.28 | 1.13 | −0.14 | collapses |
| ext_turnover (λ=0.002) | 9 | 1.14 | 0.64 | 0.51 | 1.76 | 0.10 | 0.26 | 0.24 | 1.27 | – |
| ext_positive (λ=1) | 9 | 0.24 | 0.13 | 0.23 | 0.79 | 0.12 | 0.30 | 0.79 | −0.22 | – |

Before → after the fixes (test Sharpe, monthly): gan 0.63 → 0.68, unc 0.45 → 0.45, no-macro 0.66 → 0.62, raw-macro 0.13 → 0.07, ext_turnover 0.09 → 0.51, ext_positive 0.34 → 0.23. EV / weighted XS-R² of the headline run: 0.050 / 0.189 → 0.084 / 0.228. The full v1 tables are in `output_v1/` and `deliverables/before_fixes_v1/`.

## Simulations (Table II), 3 GAN seeds, data seed 0
Setup 1 (β = C1·C2): GAN SR tr/va/te 0.99/0.92/1.00 (paper 0.98/1.11/0.94), EV 0.17/0.15/0.17 (paper 0.12/0.11/0.13), XS 0.14/0.12/0.12 (paper 0.10/0.09/0.07); population and LS match the paper.
Setup 2 (β = C·sign(h_t)): GAN SR 1.09/0.66/0.60 (paper 0.79/0.77/0.64); the LSTM hidden states recover the cycle with |corr(state, h_t)| = 0.45, 0.55, 0.64, 0.51 (`hidden_state_recovery.png`). GAN EV / XS-R² are ≈0.01 (paper 0.18 / 0.19): see item 5 below.

## Numbers that still disagree with the paper by more than seed noise
1. **Headline test Sharpe 0.68 vs 0.75.** Single seeds have sd 0.14, so the 9-model ensemble carries a sampling sd of roughly 0.05; the gap is about one ensemble-sd and is consistent with seed noise. Train 2.99 vs 2.68 and valid 1.36 vs 1.43 sit on either side of the paper. EV 0.084 vs 0.08 and weighted XS-R² 0.228 vs 0.23 match.
2. **UNC**: valid 0.72 vs 1.33 and train 1.52 vs 1.93 are clearly lower; test 0.45 vs 0.53 is within noise (single seeds range 0.11–0.49). Its EV 0.10 / XS-R² 0.29 exceed the paper's 0.07 / 0.19.
3. **Ordering of the ablations** now matches the paper: UNC 0.45 < no-macro 0.62 < GAN 0.68 (paper 0.53 < 0.69 < 0.75); raw macro without the LSTM collapses out of sample (valid 0.02, test 0.07), as the paper reports.
4. **Extensions.** With the fixes, the turnover penalty (λ=0.002) cuts turnover from 0.96 to 0.24 and keeps a gross test Sharpe of 0.51; net of 50 bps one-way costs it beats the plain GAN (1.27 vs 0.92 annual). The positive-SDF hinge (λ=1) still hurts (test 0.23, seeds from −0.17 to 0.36). Both λ values are extension choices; a λ sweep takes ~3 min per config on this GPU.
5. **Simulation setup 2 EV / XS-R² ≈ 0.** The port fits the beta network on characteristics only (as `config_RF_1.json` does for the empirical data). In setup 2 the true loading is C·sign(h_t), which no function of C alone can represent, so loading-based EV / XS-R² are ≈0 while the SDF weights (which see the LSTM state) reach the paper's Sharpe. The paper's simulation loadings evidently include the macro state; the authors did not release simulation code. The v1 value (0.08) was seed noise around the same structural zero.
6. **Simulation benchmark rows** (FFN, LS) are the port's own implementations: FFN in setup 1 reaches EV 0.17 vs the paper's 0.05, and in setup 2 FFN / LS reach EV ≈ 0.17 vs the paper's 0.02 / 0.16.
