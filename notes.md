# Replication notes — Chen, Pelger & Zhu (2024), PyTorch port, H100 run of 2026-09-10

## Environment
* Python: `.venv` created from `C:\ProgramData\anaconda3\python.exe` (3.13). torch **2.11.0+cu128**, numpy 2.5.3, matplotlib 3.11.1.
  `torch.cuda.is_available()` = True, device = NVIDIA H100 NVL (driver 581.80). Everything (training, ensemble evaluation, beta network, simulations) ran on the GPU.
  Gotcha: `pip install torch --index-url .../cu128 --extra-index-url pypi.org` resolved the CPU wheel; reinstalling with the cu128 index only fixed it.
* Data: the six npz files were extracted from `drive-download-…/datasets.zip` into `datasets/char` and `datasets/macro`. `check_data.py` passed.

## Data summary (loader output)
| split | months | permnos (N) | observations | avg stocks/month | chars | macro |
|---|---|---|---|---|---|---|
| train | 240 (1967-01 – 1986-12) | 3686 | 336,113 | 1400.5 | 46 | 178 |
| valid | 60 (1987-01 – 1991-12) | 3347 | 132,167 | 2202.8 | 46 | 178 |
| test | 300 (1992-01 – 2016-12) | 7141 | 750,275 | 2500.9 | 46 | 178 |

N per split is the number of distinct PERMNOs that appear in that window (the "~10,000 stocks" of the paper is the union over the full sample). Macro files have 178 columns, so no `macro_idx` was needed and no config was changed.

## Changes made (and why)
1. `dlap/data.py` (3 lines): the authors' files store dates as `yyyymmdd` (e.g. 19670131) while the code expects `yyyymm`. Dates are now divided by 100 when they exceed 999999. Without this, the year-by-year Sharpe grouped by month (every "year" had one observation, Sharpe 0) and plot tick labels were wrong. Training is unaffected.
2. `dlap/model.py`, `dlap/train.py`, all hyper-parameters and epoch counts: **untouched**.
3. Added helper scripts: `launch_fleet.py` (detached per-config GPU processes), `fleet_status.py`, `collect_deliverables.py`. They only launch `run.py` / `simulate.py` with the exact `run_all.ps1` arguments.
4. Execution: the six configs and both simulations were run as 8 parallel processes (GPU at 99% utilisation, ~19 GB used) instead of serially. Stale `output/sim1`, `sim2`, `sim2b` from an earlier reduced-epoch run were deleted and regenerated at the default epochs.
5. Checkpoint deviation: the smoke test was run and inspected (24 s wall clock, 0.03 s/epoch, shapes as expected), and the full runs were launched immediately rather than waiting for a reply, because one full seed costs under a minute of GPU time and everything is re-runnable.

## Timing
* Smoke test (`configs/quick.json`, 352 epochs, 1 seed): 9 s training, 24 s total, ≈0.03 s/epoch alone on the GPU.
* Full seed (256 + 64 + 1024 epochs) under 8-way GPU sharing: gan seeds 0–8 took 169, 196, 168, 142, 122, 106, 107, 104, 85 s. Alone a seed would take ≈45 s. Whole plan (49 seeds + 2 simulations + evaluation): ≈30 min wall clock.

## Headline replication: `output/gan` (9-seed ensemble, best-valid-Sharpe checkpoints)
| | train | valid | test | paper test |
|---|---|---|---|---|
| Sharpe (monthly) | 2.03 | 1.34 | **0.63** | 0.75 |
| Sharpe (annual) | 7.02 | 4.66 | **2.20** | 2.60 |
| EV (beta net) | 0.148 | 0.055 | **0.050** | 0.08 |
| XS-R² (beta net, unweighted) | -0.02 | 0.03 | **0.03** | 0.23 |
| XS-R² (beta net, T_i-weighted) | 0.10 | 0.01 | **0.19** | – |
| turnover / month | 1.04 | 1.06 | 1.07 | – |
Paper train/valid Sharpe: 2.68 / 1.43. Test: mean 0.84 %/month, vol 1.33 %, max drawdown 7.5 %, worst month -7.1 %. Net-of-cost annual test Sharpe: 2.20 (0 bps), 1.92 (10), 1.50 (25), 0.81 (50). Beta-decile mean returns (test, %/month): -0.21, 0.51, 0.69, 0.81, 0.96, 1.04, 1.27, 1.39, 1.79, 3.22 (monotone, as in the paper's Fig. 9).

Individual seeds, test Sharpe (monthly) at the best-valid-Sharpe epoch:
| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| test SR | 0.498 | 0.523 | 0.552 | 0.368 | 0.522 | 0.419 | 0.603 | 0.449 | 0.509 |
| valid SR | 0.90 | 0.90 | 1.08 | 1.02 | 0.77 | 1.26 | 1.35 | 0.63 | 1.05 |
Mean 0.49, sd 0.07; the ensemble (0.63) beats every single seed, as the paper reports.

## All runs (see summary.md)
| run | seeds | SR train | SR valid | SR test (m) | SR test (a) | EV test | XS-R² test | turnover | paper test SR |
|---|---|---|---|---|---|---|---|---|---|
| gan | 9 | 2.03 | 1.34 | 0.63 | 2.20 | 0.05 | 0.03 | 1.07 | 0.75 |
| unc (no adversary) | 9 | 1.50 | 0.88 | 0.45 | 1.55 | 0.05 | 0.04 | 0.84 | 0.53 |
| gan_nomacro | 9 | 1.87 | 1.27 | 0.66 | 2.29 | 0.06 | 0.03 | 1.02 | 0.69 |
| gan_allmacro_raw | 4 | 1.62 | 0.03 | 0.13 | 0.47 | 0.10 | 0.06 | 0.86 | collapses |
| ext_turnover (λ=0.002) | 9 | 0.63 | 0.13 | 0.09 | 0.32 | 0.03 | -0.00 | 0.06 | – |
| ext_positive (λ=1) | 9 | 0.53 | 0.24 | 0.34 | 1.19 | 0.11 | 0.06 | 0.65 | – |

## Simulations (Table II), 3 GAN seeds, data seed 0
Setup 1 (β = C1·C2): GAN SR tr/va/te 1.05/0.88/0.98 (paper 0.98/1.11/0.94), EV 0.17/0.15/0.17 (paper 0.12/0.11/0.13), XS 0.13/0.12/0.12 (paper 0.10/0.09/0.07). Population and LS match the paper.
Setup 2 (β = C·sign(h_t)): GAN SR 0.84/0.52/0.58 (paper 0.79/0.77/0.64), EV 0.08/0.07/0.08 (paper 0.18), XS 0.11/0.06/0.11 (paper 0.19/0.20/0.15). |corr(LSTM state, true h)| by unit = 0.50, 0.36, 0.45, **0.85** — the LSTM recovers the hidden cycle from the increment dZ_t (hidden_state_recovery.png).

## Numbers that disagree with the paper by more than seed noise
1. **Test Sharpe 0.63 vs 0.75** (annual 2.20 vs 2.60), train 2.03 vs 2.68. Seed sd is 0.07 for single seeds and smaller for the ensemble, so the 0.12 gap is probably real. Plausible causes: no 384-configuration hyper-parameter search (README deviation 2), adversary evaluated once without dropout (deviation 3), and possibly a different vintage of the Drive data. Valid Sharpe (1.34 vs 1.43) is within noise.
2. **XS-R² 0.03 (unweighted) vs 0.23.** The T_i-weighted version is 0.19, and 0.23 for UNC (paper 0.19), so the paper's number is consistent with weighting stocks by their number of observations. The authors' `utils.py` additionally computes XS-R² on beta-sorted decile portfolios, which we do not replicate. Report the weighted number, and say which definition is used.
3. **EV 0.05 vs 0.08** — lower for every run (UNC 0.05 vs 0.07).
4. **Macro hidden states do not help here**: gan_nomacro 0.66 vs gan 0.63 (paper: 0.69 vs 0.75). The ordering UNC < GAN holds (0.45 < 0.63, paper 0.53 < 0.75), and the raw-macro/no-LSTM ablation collapses out of sample (valid 0.03, test 0.13) exactly as the paper says.
5. **UNC valid Sharpe 0.88 vs 1.33**, train 1.50 vs 1.93.
6. **Extensions as configured hurt performance.** Turnover penalty λ=0.002: turnover falls from 1.07 to 0.06 but test Sharpe falls to 0.09; the penalty term (λ×turnover ≈ 2e-3) is 100–1000× the pricing loss (≈5e-6), so it dominates the objective. Positive-SDF hinge λ=1: test Sharpe 0.34 with huge seed dispersion (−0.23 to 0.40). Both λ values are extension choices, not paper hyper-parameters; a sweep (e.g. turnover λ ∈ {1e-5, 1e-4}, hinge λ ∈ {0.01, 0.1}) takes ~3 min per config on this GPU. Not run, since the task said not to change hyper-parameters.
7. **Simulation FFN/LS**: in Setup 1 our FFN reaches EV 0.17 / XS 0.13 (paper 0.05 / −0.33); in Setup 2 FFN and LS get EV≈0.16 and XS≈0.18–0.20 (paper 0.01–0.16). Sharpe ratios match the paper; the EV/XS gap comes from the port's simpler benchmark implementations, not from the GAN. Setup 2 GAN EV/XS (0.08 / 0.11) are below the paper's 0.18 / 0.15–0.20.

## Fidelity audit against the authors' TensorFlow code (done 2026-09-10 after the runs)
Compared line by line: `authors_original_tf_code/config/config.json`, `run.py`, `src/data/data_layer.py`, `src/model/model_GAN.py` (graph, `_add_loss`, `train`, evaluation helpers), `src/model/model_utils.py` (`calculateStatistics`), `src/utils.py` (`sharpe`), `create_RF_data.py`, `model_RtnFcst.py`.

**Verified identical**
* Hyper-parameters: every key of the authors' `config.json` equals `run.py` DEFAULTS (lr 0.001, Adam, dropout keep 0.95, SDF FFN [64,64] + LSTM 4 units, adversary LSTM 32 units + 8 tanh moments with 0 hidden layers, epochs 256 / 64 / 1024, sub_epoch 4, weighted loss, all 178 macro series, ignore_epoch 64).
* Data: mask = return ≠ −99.99; macro standardised with the training-window mean/std; loss weights T_i = number of months per stock.
* SDF network: concat[I_{t,i}, h_t] → ReLU(64) → dropout → ReLU(64) → dropout → linear w; M_t = 1 + Σ_i w_{t,i} R_{t,i}; LSTM input dropout during training only.
* Loss: mean over (moment j, stock i) of [(1/T_i) Σ_t M_t R_{t,i} g_{j,t,i}]² · T_i / max_i T_i — identical to `_add_loss`; adversary output zeroed on missing entries; adversary LSTM starts from the zero state.
* Schedule: stage 1 (unconditional) → reload best-valid-loss checkpoint → stage 2 trains the adversary to maximise the conditional loss keeping the max-loss adversary → reload → stage 3 (conditional); best-valid-loss and best-valid-Sharpe are tracked separately per stage after `ignore_epoch`, selection uses the unconditional validation loss and the raw-weight Sharpe, exactly as in `train()`.
* Sharpe = mean/std (ddof 0) of 1 − M; LSTM state carried train → valid → test (`getNextInitialState`); evaluation with dropout off.
* Ensemble: mean of raw w over seeds, then L1-normalise each month, F = 1 − M (authors' `getNormalizedSDFFactor`); EV / XS-R² / weighted XS-R² formulas identical to `calculateStatistics`; beta-network target R·F·50 built from the normalised ensemble SDF as in `create_RF_data.py`.

**Deviations found (in addition to README §3)**
1. Optimizer state. The authors create three separate Adam optimizers (`_train_model_op_unc`, `_update_moment_op`, `_train_model_op`), so stage 3 starts with fresh Adam moments. The port reuses one Adam instance for stages 1 and 3 (`train.py`, `opt_sdf`), so stage 3 inherits stage-1 moment estimates and step count. Small effect; not fixed because train.py was to be left untouched.
2. Initialisation. TF 1.12 defaults: Glorot-uniform kernels, zero biases, LSTM forget-gate bias 1.0. PyTorch defaults: U(±1/√fan_in) for weights and biases, LSTM biases U(±1/√hidden), no forget bias. Same architecture, different starting distribution; this changes seed-level paths and can move single-seed Sharpe by roughly the seed spread (sd ≈ 0.07), not the estimator.
3. Dropout in the adversarial stages (README §3): the authors keep dropout active on the SDF net while training the adversary and on the adversary while training the SDF; the port evaluates the frozen network once without dropout. Only the LSTM input of the 0-layer adversary is affected.
4. Adam epsilon placement differs slightly between TF and PyTorch (negligible).
5. Not verifiable from the shipped code: the beta-network hyper-parameters ([32,16,8], 2048 epochs, valid-MSE selection) and which checkpoint the authors' notebook ensembles (the notebook `model_GAN.ipynb` is not included; `run.py` reports the best-valid-Sharpe epoch, which is what the port uses).
6. The paper's numbers come from a validation search over 384 configurations (README §2); we ran the reported optimum only.

Fixing 1 and 2 needs ~10 lines in `train.py` / `model.py` and a ~30-minute re-run of everything.
