# Review of the PyTorch replication — Chen, Pelger & Zhu (2024)

Reviewed on 2026-09-10 against repo `dlap-torch-replication` at commit `a33eb08` (v2, "fidelity fixes and full
re-run"). Nothing was re-run and no result was altered; every number below is read out of the committed
`results.json` / `table2.json` / `log.txt` / `sdf_factor.csv`, or computed from them arithmetically.

Sources: `README.md`, `deliverables/notes.md`, `deliverables/summary.md`, the six `output/*/results.json`,
`output/sim1/table2.json`, `output/sim2/table2.json`, all PNGs under `deliverables/` and `output/*/plots/`,
and `git show a33eb08 -- dlap/`.

---

## 0. Headline verdict

The port reproduces the paper's **headline SDF and its ablation ordering**. Two things do not replicate and
should be said out loud on the slides:

1. **EV and cross-sectional R² do not rank models the way the paper's do.** The headline run matches the
   paper's *level* (EV 0.084 vs 0.08, XS-R² 0.228 vs 0.23), but every ablation — including the run that
   collapses to a test Sharpe of 0.07 — scores **equal or better** on both. In the paper these metrics
   separate the GAN from UNC. In our port they do not separate anything.
2. **The positive-SDF extension is untestable as measured.** The baseline SDF already has zero negative-M
   months, so the check the extension was built to move is vacuous at the reported scale (details in §4.4).

One factual error in `deliverables/notes.md` should be corrected before it reaches the deck: the claim that
"the 9-model ensemble (0.68) beats every single seed" is **false** — seed 0 scores 0.750 (§2).

---

## 1. (a) Our numbers next to the paper's

### 1.1 U.S. equities, test window 1992–2016 (300 months)

Sharpe ratios are monthly unless marked (a) = annualised. Our EV / XS-R² are the beta-network values;
XS-R² is the T_i-weighted one, which is the definition the authors' notebook prints (`WXSR2`) and the one
that corresponds to the paper's 0.23.

| Model | | SR train | SR valid | SR test | SR test (a) | EV test | XS-R² test |
|---|---|---|---|---|---|---|---|
| **GAN, hidden macro states** | paper | 2.68 | 1.43 | **0.75** | 2.60 | 0.08 | 0.23 |
| | **ours** (9 seeds) | **2.99** | **1.36** | **0.68** | **2.36** | **0.084** | **0.228** |
| UNC, no adversary | paper | 1.93 | 1.33 | 0.53 | 1.84 | 0.07 | 0.19 |
| | **ours** (9 seeds) | 1.52 | 0.72 | 0.45 | 1.57 | 0.104 | 0.287 |
| GAN, no macro | paper | 1.90 | 1.35 | 0.69 | 2.39 | – | – |
| | **ours** (9 seeds) | 1.75 | 1.49 | 0.62 | 2.15 | 0.101 | 0.258 |
| GAN, 178 raw macro, no LSTM | paper | 1.07 | 0.05 | 0.05 | 0.17 | – | – |
| | **ours** (4 seeds) | 1.03 | 0.02 | 0.07 | 0.23 | 0.117 | 0.279 |
| FFN forecast | paper | 0.45 | 0.42 | 0.44 | 1.52 | 0.04 | 0.15 |
| | ours | *not run* | | | | | |
| Elastic net | paper | – | – | 0.50 | 1.73 | 0.04 | 0.19 |
| | ours | *not run* | | | | | |
| Linear (OLS) | paper | – | – | 0.42 | 1.45 | 0.03 | 0.14 |
| | ours | *not run* | | | | | |

Extensions (no paper counterpart):

| Model | SR train | SR valid | SR test | SR test (a) | EV test | XS-R² test | turnover | net SR(a) @50bp |
|---|---|---|---|---|---|---|---|---|
| ext_turnover, λ=0.002 | 1.14 | 0.64 | 0.51 | 1.76 | 0.104 | 0.262 | **0.24** | **1.27** |
| ext_positive, λ=1 | 0.24 | 0.13 | 0.23 | 0.79 | 0.118 | 0.303 | 0.79 | −0.22 |
| gan (baseline, for reference) | 2.99 | 1.36 | 0.68 | 2.36 | 0.084 | 0.228 | 0.96 | 0.92 |

Headline test factor: mean 0.79 %/month, vol 1.16 %/month, worst month −4.46 %, max drawdown 5.53 %,
beta-decile spread 3.38 %/month (monotone across all ten deciles).

### 1.2 Simulation (Table II)

Setup 1, β = C₁·C₂ (pure interaction, no macro state):

| | SR train | SR valid | SR test | EV test | XS-R² test |
|---|---|---|---|---|---|
| Population (paper) | 0.96 | 1.09 | 0.94 | 0.17 | 0.17 |
| Population (ours) | 0.96 | 0.92 | 1.02 | 0.167 | 0.130 |
| GAN (paper) | 0.98 | 1.11 | 0.94 | 0.13 | 0.07 |
| **GAN (ours, 3 seeds)** | **0.99** | **0.92** | **1.00** | **0.166** | **0.123** |
| FFN (paper / ours) | 0.94 / 0.97 | 1.04 / 0.92 | 0.89 / 1.02 | 0.05 / 0.166 | −0.33 / 0.130 |
| Linear (paper / ours) | 0.07 / 0.12 | −0.10 / 0.04 | 0.01 / −0.09 | 0.00 / 0.003 | 0.01 / −0.001 |

Setup 2, β = C·sign(h_t), h_t a hidden cycle seen only through its trended increment:

| | SR train | SR valid | SR test | EV test | XS-R² test |
|---|---|---|---|---|---|
| Population (paper) | 0.89 | 0.92 | 0.86 | 0.17 | 0.15 |
| Population (ours) | 0.97 | 0.93 | 0.97 | 0.168 | 0.203 |
| GAN (paper) | 0.79 | 0.77 | 0.64 | 0.17 | 0.15 |
| **GAN (ours, 3 seeds)** | **1.09** | **0.66** | **0.60** | **0.006** | **0.004** |
| FFN (paper / ours) | 0.05 / 0.06 | −0.05 / 0.11 | 0.06 / 0.02 | 0.02 / 0.162 | 0.02 / 0.198 |
| Linear (paper / ours) | 0.12 / 0.06 | −0.05 / 0.11 | 0.10 / 0.02 | 0.15 / 0.168 | 0.14 / 0.203 |

LSTM hidden-state recovery in setup 2: |corr(state, h_t)| = 0.45, 0.55, 0.64, 0.51 across the four units.

---

## 2. (b) Per-seed dispersion of the headline run

Test Sharpe (monthly) at each seed's best-validation-Sharpe checkpoint, `output/gan/log.txt`:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| test | **0.750** | 0.378 | 0.478 | 0.589 | 0.481 | 0.600 | **0.283** | 0.394 | 0.458 |
| valid | 0.97 | 0.88 | 0.92 | 1.02 | 1.00 | 1.02 | 1.00 | 0.72 | 1.21 |
| train | 1.98 | 2.85 | 3.43 | 2.91 | 2.00 | 2.91 | 3.68 | 3.14 | 4.30 |

**Range 0.283 – 0.750; mean 0.490; sd 0.139. Ensemble 0.681.**

The single-seed spread (0.47 wide) is more than half the paper's headline number. That is the strongest
argument on the deck: any single-seed comparison in this literature is noise.

**Correction to `notes.md`.** It states the ensemble "beats every single seed, as the paper reports". It does
not: seed 0 (0.750) beats the ensemble (0.681). The same is true for UNC (best seed 0.494 vs ensemble 0.453)
and for the no-macro run (0.657 vs 0.620). The defensible statement, which is what the deck now says, is
that **the ensemble beats the average seed by 0.19 and beats 8 of 9 seeds**. Only `ext_turnover` has an
ensemble (0.509) above all of its seeds (max 0.350).

---

## 3. (c) Internal-consistency checks

| # | Check | Result |
|---|---|---|
| 1 | SR train ≥ valid ≥ test | **Passes** for gan (2.99 ≥ 1.36 ≥ 0.68), unc, gan_nomacro, ext_turnover. **Fails** for gan_allmacro_raw (valid 0.019 < test 0.068) and ext_positive (valid 0.130 < test 0.229). |
| 2 | Adversary > no adversary | **Passes.** 0.681 vs 0.453 on test (paper 0.75 vs 0.53). |
| 3 | LSTM states > no macro > raw macro | **Passes on test Sharpe.** 0.681 > 0.620 > 0.068 (paper 0.75 > 0.69 > 0.05). **Fails on validation Sharpe**: no-macro 1.490 > GAN 1.358, where the paper has 1.43 > 1.35. |
| 4 | Same ranking on EV / XS-R² | **Fails, and this is a real finding.** See §4.1. |
| 5 | Net-of-cost SR monotone decreasing in cost | **Passes in all six runs** at 0/10/25/50 bps. Baseline 2.36 → 2.07 → 1.64 → 0.92; penalised 1.76 → 1.67 → 1.52 → 1.27. |
| 6 | Higher turnover ⇒ steeper cost decay | **Passes where comparable.** Turnover 0.96 loses 1.44 annual SR over 0→50 bps; turnover 0.24 loses 0.49. (Not comparable for gan_allmacro_raw: its cost drag looks small only because its factor vol is 5.1 %/month.) |
| 7 | Beta-decile returns monotone in decile | **Passes** for gan, unc, gan_nomacro, ext_turnover (10/10 monotone). **Fails at decile 3** for gan_allmacro_raw and ext_positive — both degenerate runs. |
| 8 | Positive-SDF run has fewer negative-M months than baseline | **Vacuous — cannot pass or fail.** See §4.4. |

The two runs that fail checks 1 and 7 are the same two runs: the raw-macro ablation (which the paper also
reports as collapsing) and the positive-SDF extension. Both failures are symptoms of a degenerate fit, not
of an evaluation bug: gan_allmacro_raw has a 93.5 % max drawdown and a 5.1 %/month factor vol.

---

## 4. (d) Disagreements with the paper beyond seed noise

### 4.1 EV and XS-R² do not discriminate between models — *the most important finding*

| run | test SR | EV (beta net) | XS-R² (weighted) |
|---|---|---|---|
| gan | **0.68** | 0.084 | 0.228 |
| unc | 0.45 | 0.104 | 0.287 |
| gan_nomacro | 0.62 | 0.101 | 0.258 |
| gan_allmacro_raw | **0.07** | 0.117 | 0.279 |
| ext_turnover | 0.51 | 0.104 | 0.262 |
| ext_positive | 0.23 | **0.118** | **0.303** |

Test Sharpe varies 10-fold across these runs; EV moves inside 0.084–0.118 and XS-R² inside 0.228–0.303, and
both are **inversely** related to Sharpe. The run with the *worst* SDF has the *highest* EV. In the paper the
GAN dominates UNC on both (0.08/0.23 vs 0.07/0.19).

Most likely cause: the second-stage beta network is fitted on the 46 characteristics only (as the authors'
`config_RF_1.json` does), with a target of R·F·50. Its own R² against realised returns is largely driven by
what characteristics explain about returns, not by which SDF factor F it was pointed at. Once F is any
reasonable long-short factor, the projection explains a similar amount. So the headline EV/XS-R² match
(0.084/0.228 vs 0.08/0.23) is real but should not be presented as independent confirmation.

### 4.2 The XS-R² match depends on a weighting choice

The same run gives XS-R² = **0.228** T_i-weighted and **0.045** unweighted — a factor of five. Only the
weighted one matches the paper's 0.23. The choice is justified (the authors' notebook prints `WXSR2`) and is
documented in `notes.md`, but a reader should be told that "we match 0.23" is conditional on it.

### 4.3 Headline Sharpe 0.68 vs 0.75

Gap = 0.07 against a per-seed sd of 0.139; a nine-model ensemble carries a sampling sd of roughly 0.05, so
the gap is about 1.5 ensemble-sd. Consistent with seed noise, but on the low side, and it is one-sided:
train came in *above* the paper (2.99 vs 2.68) while test came in below, which is the signature of a fit that
went slightly further into the training window before validation stopped it. Six of nine seeds pick an epoch
in the last 25 of stage 3 (of 1024), i.e. validation Sharpe was still improving when the schedule ended.
The honest reading for the slide: **we land ~9 % below the paper's headline on one-fortieth of its compute,
with no hyper-parameter search.**

### 4.4 The positive-SDF extension is untestable from the released outputs

`sdf_factor.csv` stores F on the **L1-normalised** ensemble weights, so M = 1 − F lies in [0.944, 1.045] on
the test window. Counts:

| run | negative-M months (test) | min M |
|---|---|---|
| gan (baseline) | **0 / 300** | 0.9444 |
| ext_positive (λ=1) | **0 / 300** | 0.9461 |

The baseline is already at zero, so "fewer negative-M months" cannot be demonstrated. The hinge
λ·E[max(0,−M)²] acts during training on M built from the **raw** (unnormalised) weights, where M can be far
from 1 — but `weights_*.npy` stores the *normalised* weights (`run.py:108` saves `wn`), and the datasets are
gitignored (licensed CRSP), so the raw-weight M cannot be recomputed. Net effect: the extension costs
two-thirds of the test Sharpe (0.68 → 0.23) and buys nothing measurable. That is a legitimate result to
present, but it must be framed as "the constraint is not binding at the scale we report M", not as "the
penalty didn't help".

To make this extension testable, `run.py` would need to also save the raw ensemble weights, or the per-month
L1 norm, alongside the normalised ones. Recommended for 230ZB.

### 4.5 UNC train and validation are well below the paper

Ours 1.52 / 0.72 vs paper 1.93 / 1.33. Test (0.45 vs 0.53) is inside the seed range (0.112–0.494). One cause
is visible in the log: **seed 6 of the UNC run collapsed**, selecting epoch 3 with SR 0.092/0.046/0.112,
against epochs 220–255 for the other eight. It is inside the 9-model ensemble. The paper does not say how it
handles a degenerate seed.

### 4.6 Simulation setup 1: our benchmarks are stronger than the paper's

Our GAN matches the paper on Sharpe (1.00 vs 0.94 test) but scores EV 0.166 / XS-R² 0.123 against the paper's
0.13 / 0.07 — i.e. we sit essentially *on the population* (0.167 / 0.130) where the paper's GAN sits below it.
Our FFN benchmark reaches the same 0.166 / 0.130, where the paper's FFN gets 0.05 / **−0.33**. The authors
released no simulation code, so the FFN and LS rows are this repo's own implementations. Read as: our
benchmark FFN is simply better-implemented than theirs, not that the GAN failed to replicate. Worth
mentioning if asked, not worth a slide.

### 4.7 Simulation setup 2: EV and XS-R² are structurally zero, and the DGP calibration differs

Our GAN gets EV 0.006 / XS-R² 0.004 against the paper's 0.17 / 0.15. `notes.md` explains this correctly: the
beta network sees characteristics only, the true loading is C·sign(h_t), and no function of C alone can
represent it, so loading-based metrics are ≈0 by construction while the SDF weights (which do see the LSTM
state) still reach the paper's Sharpe. Two things `notes.md` does not say:

* Our **population** SDF in setup 2 scores SR 0.97 test against the paper's 0.86, so the two DGP calibrations
  are not identical and the level comparison is not apples-to-apples. Relative to its own population, our GAN
  captures 0.60/0.97 = 62 % of the attainable Sharpe against the paper's 0.64/0.86 = 74 %.
* Our LS benchmark reaches EV 0.168 / XS-R² 0.203 — essentially the population — which again reflects our
  benchmark implementation, not the paper's.

### 4.8 Variable importance is nearly flat

`output/gan/variable_importance.json`: ST_REV 0.085, SUV 0.056, then **43 of 46 characteristics packed
between 0.050 and 0.038**. The paper's Figure 8 shows a far more concentrated profile. So "which
characteristics the SDF uses" is *not* replicated; only the top two separate from the pack. Say this on the
slide rather than claiming a match.

### 4.9 The LSTM macro states do not visibly trace the business cycle

`macro_states_test.png` shows four high-frequency series with no obvious cyclical structure and no visible
alignment with 2001 or 2008, unlike paper Figure 13. This **cannot be quantified from the released outputs**:
the state series are plotted but never written to disk, and the macro npz files are gitignored. The deck's
original caption ("[ ] of 4 line up with 2001 and 2008") has therefore been rewritten to describe what the
picture actually shows instead of asserting a count. See §6.

### 4.10 The empirical benchmarks were not replicated

FFN forecast, elastic net and OLS on U.S. equities are quoted from the paper, never run. Only the simulation
has our own FFN/LS. The deck labels those rows "(paper)" and says so out loud — that is the honest handling,
but expect the question.

### 4.11 Coverage gap: only 4 seeds for the raw-macro ablation

`gan_allmacro_raw` uses 4 seeds where everything else uses 9, and two of those four selected epoch 0 or 1
(SR 0.138/0.048/0.029 and 0.207/−0.045/−0.022), i.e. they never trained. The conclusion (collapse) matches
the paper and is not in doubt, but the number 0.07 is fragile.

---

## 5. Code audit: `data.py`, `model.py`, `train.py` vs `notes.md`

Diffed `f9a02cc` (v1) → `a33eb08` (v2, the reported numbers). **Both `model.py` and `train.py` were changed,
substantively.** Every hunk maps onto a documented item; no undocumented change was found in any of the three
files, and nothing touches the evaluation formulas or the reported metrics.

| File | Change | `notes.md` item | Verdict |
|---|---|---|---|
| `data.py` | Macro standardisation moments computed in float64, cast to float32 after | 8 | Documented, matches the authors' data layer |
| **`model.py`** | **New** `init_tf_default_`: Glorot-uniform kernels, zero biases, LSTM Glorot over the joint [input+hidden, 4·hidden] kernel, forget-gate bias 1 | 1 | Documented. Applied to `SDFNet`, `MomentNet` and (via `evaluate.py`) `BetaNet` |
| **`model.py`** | **New** `TFAdam`: `tf.train.AdamOptimizer` update with ε inside the bias-corrected denominator | 2 | Documented. Verified the implementation does not corrupt the second-moment state (`v.sqrt()` allocates before `add_`) |
| **`train.py`** | Stage 2 adversary loop: single 64-step pass → 4 sub-epoch passes of 64 with the running max reset each pass | 4 | Documented |
| **`train.py`** | Stage 2/3: frozen network evaluated **once with dropout off** → re-evaluated **every step with dropout on** | 3 | Documented. This is the largest behavioural change of the eight |
| **`train.py`** | One shared Adam for stages 1 and 3 → three fresh `TFAdam` instances (`opt_unc`, `opt_adv`, `opt_cond`) | 2 | Documented |
| **`train.py`** | `ignore_epoch` default 64 → 32 | 5 | Documented (also changed in `run.py` DEFAULTS) |
| `evaluate.py` | Beta net validation checked every 16 epochs → every epoch; `TFAdam`; TF init | 6 | Documented |
| `run.py` | Variable importance: seed-0, δ=1e-4 → 9-model average, δ=1e-6 | 7 | Documented |
| `summarize.py` | Reports the T_i-weighted XS-R² first | — | Presentation only; consistent with §4.2 |

The extension code (`turnover_penalty`, `sdf_hinge`, `pricing_loss`) is **unchanged** between v1 and v2, and
both extensions are off in `configs/gan.json`, so the headline run is a pure replication.

Effect of the fixes on the test Sharpe (from `notes.md`, cross-checked against `output_v1/`):
gan 0.63 → 0.68, unc 0.45 → 0.45, no-macro 0.66 → 0.62, raw-macro 0.13 → 0.07, ext_turnover 0.09 → **0.51**,
ext_positive 0.34 → 0.23. EV/XS-R² of the headline run 0.050/0.189 → 0.084/0.228. The fixes moved the
headline toward the paper; they moved `ext_turnover` by 5×, which is worth knowing because the extension
conclusion in §7 rests on it.

---

## 6. (e) Which plots are slide-ready

**Use as-is**

| Plot | Why |
|---|---|
| `output/sim2/hidden_state_recovery.png` | The best picture in the repo. Two stacked panels, the recovered cycle is unmistakable. → simulation slide. |
| `output/gan/plots/training_curves.png` | Shows the three-stage schedule, the stage-2 discontinuity, and train/valid divergence in one image. → method slide. **Was placed at the wrong aspect ratio (5.85 × 2.85 for a 3.06:1 image); fixed.** |
| `output/gan/plots/cumulative_sdf_factor.png` | Clean, split cuts marked, reads at any size. → "what the model learned". |
| `output/gan/plots/beta_deciles_test.png` | Clean and monotone; good backup slide for a question about EV. |
| `output/gan_allmacro_raw/plots/cumulative_sdf_factor.png` | Excellent "collapse" picture: grows through train, flat then drawdown after 1992. Good backup for the ablation question. |

**Usable with care**

| Plot | Problem | Handling |
|---|---|---|
| `variable_importance_test.png` | 46 rotated x-labels at 7 pt. At 4 in wide they render at ~2.8 pt — unreadable projected. | Placed at 6.0 in wide (≈4.2 pt) and the caption now makes the *shape* the point, not the names. For 230ZB, re-plot the top 15 from `variable_importance.json` (no re-training needed). |
| `macro_states_test.png` | No legend, no recession shading, very noisy; does not support the claim the original caption made. | Kept, caption rewritten to §4.9. |
| `sharpe_by_year_test.png` | Fine, but 1995 (SR 13.0) compresses everything else. | The deck uses a native pptx chart with the same 25 values so the styling matches; either is defensible. |

**Do not use**

| Plot | Why |
|---|---|
| `output/quick/plots/*` | Smoke-test run (352 epochs, 1 seed), not a result. |
| `output/gan_allmacro_raw/plots/macro_states_test.png` | That config has `use_rnn: false`, so these are not LSTM states and it is not comparable to Figure 13. |
| `deliverables/before_fixes_v1/**` | v1, superseded. Keep for the before/after story only. |
| `output/gan_nomacro/plots/macro_states_test.png` | Does not exist, correctly — that config has `macro_feature_dim: 0`. Not an error. |

---

## 7. What the extensions actually show

**Turnover penalty (λ = 0.002).** Cuts test turnover 0.96 → 0.24 and gross annual Sharpe 2.36 → 1.76. Net of
cost:

| one-way cost | baseline | with penalty |
|---|---|---|
| 0 bps | **2.36** | 1.76 |
| 10 bps | **2.07** | 1.67 |
| 25 bps | **1.64** | 1.52 |
| 50 bps | 0.92 | **1.27** |

**The penalty is not a free win: it loses at 25 bps and wins at 50.** Linear interpolation puts the crossover
at **≈31 bps one-way**. That is the honest headline, and it is a better slide than "costs matter" because it
gives the audience a number to argue with. (Caveat: λ = 0.002 was a single choice, not a sweep; `notes.md`
says a sweep costs ~3 min per config.)

**Year-by-year Sharpe (230P improvement 1).** First five test years mean 6.15 annualised, last five 2.42;
1992–2004 mean 4.72 vs 2005–2016 mean 2.43; two negative years (2003, −0.13; 2009, −0.27); median 3.37. The
decay is real and is the strongest available argument for the re-estimation point made in 230P — the paper
reports one number for all 25 years.

**Positive SDF (λ = 1).** See §4.4 — presented as "the constraint is not binding at the scale we report M".

---

## 8. Numbers left unfilled in the deck

One, handled by rewriting rather than by leaving a gold placeholder (the deck now has no `[ ]` left):

| Deck text (original) | Status |
|---|---|
| Slide 9, macro-states caption: "Ours: **[ ]** of 4 line up with 2001 and 2008." | **Not computable.** The LSTM state series are plotted but never written to disk, and `datasets/macro/*.npz` is gitignored. Caption rewritten to describe the picture (§4.9). To make it computable, `run.py`'s `macro_states()` would need to dump the states to CSV. |

Everything else on the deck is filled from `results.json`, `table2.json`, `variable_importance.json`,
`sdf_factor.csv` or the run logs. Two deck rows are labelled "(paper)" and were never run by us — the
empirical FFN and elastic-net benchmarks (§4.10).

---

## 9. Notes on the deck build

Everything lives in `deck/` in this repo: `230ZA_Presentation_Deck.pptx`, a PDF export, `build_deck.js`,
`make_charts.py`, `charts/`, `review.md`, `script.md`, `README.md`. 14 slides, speaker notes on all 14,
budgeted at 10:35 of speaking against a 15-minute slot.

Three changes worth flagging because they were not in the original `build_deck.js`:

1. **The two extension charts are now PNGs, not native pptx charts.** pptxgenjs writes real OOXML chart parts
   and they were present and correct in the file — but only PowerPoint renders them. Keynote drops them
   silently, and so do Google Slides and most PDF converters, leaving a blank rectangle where the chart
   should be. Both extension slides were built around a chart, so `make_charts.py` now renders them from the
   same `results.json` values in the deck palette, and `build_deck.js` places them as images. Same numbers,
   renders everywhere.
2. **Two slide titles were shortened** because they wrapped to a second line at 32 pt and pushed into the
   content below (slides 8 and 9). The image on slide 5 was also placed at 5.85 × 2.85 in for a 3.06:1 PNG —
   visibly stretched — and is now at its true aspect.
3. **The appendix slide's first row was wrong.** It described the port as computing the adversarial moments
   once with dropout off, which was true of v1 and was fixed in v2 (`notes.md` item 3). Left as written it
   would have contradicted the numbers on the deck. The table now lists only what still differs on purpose.

Rendering: LibreOffice is not installed on this machine and PowerShell COM is Windows-only, so the PDF was
exported through Keynote. Every slide was inspected at 110 dpi: no overflowing text, no wrapped titles, no
stretched images, no gold `[ ]` left, and `pdftotext` finds none of the original scaffolding strings
(`[ ]`, `illustrative`, `Replace with`, `Insert`, `Fill from`, `PLOTS/`, `[Names]`).
