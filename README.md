# Deep Learning in Asset Pricing — PyTorch replication (MFE 230ZA)

PyTorch port of Chen, Pelger & Zhu, *Deep Learning in Asset Pricing* (Management Science 2024), built to run
on the authors' own data files and to reproduce Table II (simulation) and Table III (U.S. equities).

The original code (`jasonzy121/Deep_Learning_Asset_Pricing`) is TensorFlow 1.12 / Python 3.6 and no longer installs on a
modern machine; this port follows it line-by-line (same data format, same three-stage GAN schedule, same loss weighting,
same evaluation formulas), and adds the two extensions we proposed in 230P as config switches.

## 0. One-time setup on the H100 box (≈10 min)

```bash
git clone <this folder>  # or scp it
cd dlap_torch
pip install torch numpy matplotlib gdown           # torch with CUDA – check: python -c "import torch;print(torch.cuda.is_available())"

# authors' data + sample checkpoints (Google Drive folder linked from their README)
gdown --folder "https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l" -O drive_dl
# put the npz files where the configs expect them:
mkdir -p datasets/char datasets/macro
find drive_dl -name "Char_*.npz"  -exec cp {} datasets/char/  \;
find drive_dl -name "macro_*.npz" -exec cp {} datasets/macro/ \;
ls datasets/char datasets/macro     # expect Char_{train,valid,test}.npz and macro_{train,valid,test}.npz
```
If `gdown` fails on the folder (Drive sometimes blocks large folders), open the link in a browser, download
`datasets.zip` manually and `scp` it up.

Sanity-check the install without data (30 s): `python run.py --synthetic --logdir output/synth`

## 1. What to run, in priority order

```bash
# (a) 2 min – confirm the real data loads and the shapes match the paper (T=240/60/300, ~10k permnos, 46 chars, 178 macro)
python run.py --config configs/quick.json --logdir output/quick --seeds 1

# (b) headline replication: paper config, 9 seeds ensemble  (~5–10 min/seed on H100 → under 1.5 h)
python run.py --config configs/gan.json --logdir output/gan --seeds 9

# (c) ablations (Figure 6 / Table 1 of the paper)
python run.py --config configs/unc.json            --logdir output/unc            --seeds 9   # no adversary
python run.py --config configs/gan_nomacro.json    --logdir output/gan_nomacro    --seeds 9   # characteristics only
python run.py --config configs/gan_allmacro_raw.json --logdir output/gan_allmacro_raw --seeds 4  # 178 raw macro, no LSTM (paper: collapses)

# (d) our extensions
python run.py --config configs/ext_turnover.json     --logdir output/ext_turnover --seeds 9  # cost-aware SDF (penalty inside the loss)
python run.py --config configs/ext_positive_sdf.json --logdir output/ext_positive --seeds 9  # admissible (positive) SDF

# (e) one table for the slides
python summarize.py output/gan output/unc output/gan_nomacro output/gan_allmacro_raw output/ext_turnover output/ext_positive

# (f) simulation (Table II) – runs anywhere, ~3 min/seed on GPU
python simulate.py --setup 1 --seeds 3
python simulate.py --setup 2 --seeds 3
```
`run_all.sh` chains (b)–(e). Seeds run sequentially; to use the GPU better, launch several `python run.py ... --seed_start k --seeds 1`
in parallel (the H100 has plenty of memory: the full training panel is ~1M rows × 50 features) and then re-run with `--eval_only`.

Any config key can be overridden on the command line, e.g. `--set num_epochs=512 turnover_penalty=0.005`.

## 2. Outputs (per logdir) and where they go in the deck

| file | content | slide |
|---|---|---|
| `results.json` | SR (monthly & annual), EV, XS-R² (both w-projection and beta-network), turnover, net-of-cost SR at 0/10/25/50 bps, Sharpe by calendar year, max drawdown, beta-decile returns — for train/valid/test | results, extensions |
| `plots/cumulative_sdf_factor.png` | cumulative SDF-factor return with train/valid/test cuts | results |
| `plots/training_curves.png` | Sharpe & loss by epoch, shows the 3-stage schedule | method / challenges |
| `plots/macro_states_test.png` | the 4 LSTM hidden states over 1992–2016 (paper Fig. 13) | results |
| `plots/variable_importance_test.png` | avg \|∂w/∂characteristic\| ranking (paper Fig. 8) | results |
| `plots/sharpe_by_year_test.png` | annualised SR per year (our 230P "report it year by year" point) | extensions |
| `plots/beta_deciles_test.png` | mean return of SDF-beta sorted deciles (paper Fig. 9) | results |
| `sdf_factor.csv`, `weights_*.npy` | the SDF factor and normalised weights | appendix |
| `output/sim*/table2.json`, `hidden_state_recovery.png` | Table II replication, LSTM recovering the hidden cycle (paper Fig. 5) | replication |

Paper reference numbers (test window 1992–2016): GAN SR 0.75 monthly = 2.6 annual, EV 0.08, XS-R² 0.23; FFN 0.44/0.04/0.15;
EN 0.50/0.04/0.19; LS 0.42/0.03/0.14. Train SR of GAN 2.68, valid 1.43.

## 3. Deliberate deviations from the authors' code (say these on the "challenges" slide)

1. **Framework.** TF 1.12 graph code → PyTorch 2. Placeholders / `dynamic_rnn` / `boolean_mask` rewritten; LSTM state is carried
   train → valid → test by running the LSTM over the concatenated macro history (identical to their `getNextInitialState`).
2. **Hyper-parameter search.** The paper fits 384 configurations × validation selection × 4 finalists × 9 seeds. We take their
   reported optimum (Table I) and run 9 seeds — same architecture, one tenth of the compute.
3. **Beta network.** Kept as a separate second stage exactly as in `create_RF_data.py` + `config_RF/config_RF_1.json`
   (target `R·F·50`, FFN [32,16,8], keep 0.95, 2048 single-step epochs, best-valid-MSE checkpoint, 9 trials averaged).
   EV / XS-R² are reported both from the beta network (paper) and from projecting on w directly. The paper's XS-R² is the
   T_i-weighted one (the authors' notebook prints `WXSR2`), which `summarize.py` puts first.
4. **Extensions are opt-in flags** (`turnover_penalty`, `sdf_hinge`), so `configs/gan.json` is a pure replication.

Everything else is matched to the TF graph after a line-by-line audit (see `notes.md`): TF-default initialisation
(Glorot-uniform kernels, zero biases, LSTM forget bias 1), a `tf.train.AdamOptimizer`-exact update with a fresh optimizer
per stage, dropout kept active on the frozen network during both adversarial stages, the adversary's 4 × 64 steps with the
running-max reset of the authors' loop, `ignoreEpoch = 32` from the authors' notebook command, and the 9-model finite-difference
variable importance with delta 1e-6.

## 4. Extensions implemented

* **Cost-aware SDF** — `turnover_penalty` λ adds λ·E_t Σ_i |w̃_{t,i} − w̃_{t−1,i}| to the pricing loss (w̃ = L1-normalised
  weights, same PERMNO). Post-hoc net-of-cost Sharpe at several bps is computed for *every* run, so the comparison is
  "penalise inside the objective" vs "subtract costs after the fact" (our 230P Improvement 2).
* **Admissible SDF** — `sdf_hinge` λ adds λ·E[max(0, −M_t)²]; the paper checks positivity in-sample but never enforces it.
* **Year-by-year Sharpe** in the 25-year test window (our 230P Improvement 1 asked for exactly this breakdown).
* Ready hooks for more: `macro_idx` lets you drop the 46 characteristic medians or the Goyal–Welch series; `num_condition_moment`
  stresses the adversary (D = 4/8/16); `hidden_dim`/`num_layers` re-run the Table I grid.
