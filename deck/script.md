# Speaker script — MFE 230ZA, Group 14

**15 minutes including questions. Slides budgeted at 10:35, leaving ~4:25 for Q&A.**
The same text is in the pptx speaker notes. Times in brackets are per-slide budgets, not clock times.

| slide | budget | running |
|---|---|---|
| 1 Title | 0:20 | 0:20 |
| 2 Recap | 0:55 | 1:15 |
| 3 Pipeline | 0:40 | 1:55 |
| 4 Data | 0:50 | 2:45 |
| 5 Method | 0:50 | 3:35 |
| 6 Challenges | 1:15 | 4:50 |
| 7 Simulation | 0:55 | 5:45 |
| 8 Results, U.S. equities | 1:40 | 7:25 |
| 9 What the model learned | 0:45 | 8:10 |
| 10 Extension: costs | 1:05 | 9:15 |
| 11 Extension: stability + admissibility | 0:55 | 10:10 |
| 12 Next | 0:25 | 10:35 |
| 13 Questions | — | — |
| 14 Appendix | backup | — |

If you are running long, the two slides to compress are **3 (pipeline)** and **9 (pictures)**. Never compress
slide 8. If you are running very long, drop slide 3 entirely — slide 5 carries the method.

---

## 1 · Title — 0:20

Last term we wrote a referee report on Chen, Pelger and Zhu. This term we ran it.

Everything you are about to see comes out of a PyTorch port of their TensorFlow code, on their own data, on
an H100. The interesting part is not that it worked — it is the two places where it did not.

---

## 2 · Recap — 0:55

*The professor asked for at most a minute here. Do not re-explain the model.*

Left card, fast: three networks. An LSTM that compresses 178 macro series into four states, a feedforward net
that produces the SDF weights, and an adversary that picks the hardest test assets. The loss is the
no-arbitrage condition E[M R] = 0, not a forecast error. Headline: out-of-sample Sharpe 2.6 annual.

Our two objections were: it is fitted once in 1986 and then reported as a single number for twenty-five
years, and it turns over essentially the whole book every month with every Sharpe quoted gross of costs.

Right card: today. We replicated it end to end — simulation and U.S. equities. We ported the dead
TensorFlow-1.12 code. We ran the ablations. And we built both of last term's objections into the objective,
plus a third extension the paper skips.

Move on.

---

## 3 · Pipeline — 0:40

*Point at the boxes. Do not read them.*

Left to right: their data panels, then three training stages, then the ensemble, then the metrics.

Two things worth saying out loud. Stage 2 is where the adversary picks the assets that are hardest to price
— that is the whole idea of the paper. And everything downstream is a nine-seed ensemble, which turns out to
matter more than we expected. That comes back on slide 8.

The strip at the bottom is the complete list of what we reproduced.

---

## 4 · Data — 0:50

The authors share pre-built npz panels, and that is both the good news and the bad news.

Good news: the shapes match the paper on the first load. 240, 60 and 300 months; 46 characteristics; 178
macro series; 1.2 million stock-months. Nothing had to be reconstructed.

Bad news, right-hand card, three gaps. No PERMNOs, so we cannot join anything new to this panel. No raw CRSP
or Compustat, so the characteristics cannot be rebuilt and the sample cannot be extended past 2016. And the
macro data is final-vintage and pre-transformed, so our real-time critique from 230P cannot even be tested
from this file.

One surprise worth a sentence: each split holds three to seven thousand stocks, not ten thousand. The ten
thousand in the paper is the union over the whole sample.

---

## 5 · Method — 0:50

The objective, in one line. This is the only equation in the deck.

Read it as: for every test asset j and every stock i, the average of M times the excess return times the
adversarial weight should be zero. Stage 1 sets g to one — that is the unconditional model. Stage 2
maximises this in g. Stage 3 minimises it in the SDF weights with g frozen.

Selection card: nothing in the 1992–2016 window touches any decision. Test Sharpe is printed every epoch only
so we can draw the picture on the right.

And that picture is the honest story of the method. At the dashed line the network is reset to its best
validation weights and the adversarial moments take over. Training Sharpe then runs away to 3.0 while
validation and test sit flat at 0.7. The ensemble is what keeps that gap honest.

The table says our hyper-parameters are theirs.

---

## 6 · Challenges — 1:15

*This is the slide the assignment is really about. Slow down.*

**One.** The code is dead. TensorFlow 1.12 on Python 3.6 does not install in 2026 — placeholders,
`dynamic_rnn`, `boolean_mask`, none of it. And before any of that, a plain `pip install` gave us the CPU
build of torch. The H100 sat idle until we reinstalled from the CUDA index alone.

**Two, and this is the one worth telling.** Our first complete run looked right. Right shapes, right
pictures, ablations in the right order. And it was wrong: test Sharpe 0.63 against the paper's 0.75, and our
cost-aware extension came out at 0.09, basically useless. Nothing was broken. PyTorch simply has different
defaults from TensorFlow — different initialisers, epsilon in a different place inside Adam, and dropout
switched off on a frozen network where TensorFlow feeds keep-probability to the whole graph. Eight fixes
later the headline went to 0.68 and the extension went to 0.51. A five-fold move from changes that are
completely invisible in a results table. If you take one thing away from this replication, take that.

**Three.** We could not redo their 384-configuration search, so we took their reported optimum and spent the
compute on seeds instead — about one fortieth of their budget.

**Four.** The LSTM state has to carry across the split boundary, or the model prices January 1992 with
amnesia.

---

## 7 · Simulation — 0:55

Before the real data: does the port work at all? The simulation is where you can answer that, because there
is a population SDF to compare against.

Setup 1, top table. Our GAN gets test Sharpe 1.00. The population SDF gets 1.02. The paper gets 0.94. And the
linear model gets minus 0.09 — it cannot see an interaction between two characteristics, which is exactly the
point the paper is making. So the port is right.

Setup 2, bottom table. Here the loading flips sign with a hidden macro cycle that is only observable through
a noisy, trended increment. Our Sharpe is 0.60 against the paper's 0.64.

The picture on the right is the one to point at. Top panel is the true hidden state. Bottom panel is what the
LSTM reconstructed from the increment alone — correlations of 0.45 to 0.64 across the four units. That is the
mechanism the whole macro half of the paper rests on, and it works.

*If asked about the two zeros in the bottom table:* our β-network is fitted on characteristics only, exactly
as the authors' released config does it, and the true loading depends on the macro state. So those two
metrics are zero by construction, not by failure. The Sharpe, which uses the weights, is fine.

---

## 8 · Results, U.S. equities — 1:40

*The main slide. Give it time.*

Row 1 is the paper, row 2 is us. 0.68 against 0.75 monthly; 2.36 against 2.60 annual. We are 9 % low.

The most likely reason is visible in the same row: our training Sharpe is 2.99 against their 2.68. Our fit
ran a bit further into the training window before validation stopped it, and we did none of their
384-configuration search. EV of 0.084 and cross-sectional R² of 0.228 sit right on their 0.08 and 0.23.

Then the ablations, and this is the part that replicates cleanly. Take the adversary away and it falls to
0.45. Take the macro states away and it falls to 0.62. Feed all 178 raw macro series without the LSTM and it
collapses to 0.07 out of sample. The paper reports the same ordering.

**Middle card — the honest finding.** That ordering only replicates on Sharpe. On EV and cross-sectional R²
the ablations all score as high as the headline model, or higher. The run that collapses to a Sharpe of 0.07
has the *best* EV in the table. Our second-stage β-network is fitted on characteristics alone, so it explains
a similar amount of return whatever SDF factor you point it at. We would not present EV as evidence that this
model beats the alternatives.

**Right card — the number we would put in a referee report.** Across nine seeds the test Sharpe runs 0.28 to
0.75. The paper's own gap between the GAN and OLS is smaller than the spread between two seeds of the same
model. The ensemble at 0.68 beats the average seed at 0.49, and it beats eight of the nine — but not seed 0,
which came in at 0.75.

---

## 9 · What the model learned — 0:45

Three pictures, fifteen seconds each.

**One**, cumulative SDF factor. The dashed lines are the split boundaries. It keeps climbing after 1992, and
the worst month in twenty-five years is minus 4.5 % with a 5.5 % maximum drawdown. That is the picture that
sells the paper.

**Two**, variable importance. Short-term reversal and SUV separate from the pack; then forty-three of
forty-six characteristics sit in a flat band between 0.050 and 0.038. The paper's Figure 8 is far more
concentrated. So we reproduce the model's performance without reproducing its story about which
characteristics matter. Do not over-claim here.

**Three**, and this one does not replicate. Their Figure 13 shows four smooth states that peak in recessions
the network never saw — that is the headline picture of the macro half of the paper. Ours are noise at
monthly frequency. And we cannot even quantify it, because the state series are plotted and then thrown away
rather than saved. That is on our list to fix.

---

## 10 · Extension 1: transaction costs — 1:05

Our first 230P objection, now inside the model.

The number on the left is the one we were guessing at last term: monthly turnover of 0.96. We estimated about
1.0 from their figures. So the criticism was right — this book turns over essentially in full every month.

How bad is it? At 25 basis points one-way the headline 2.36 becomes 1.64. It survives. At 50 basis points it
is 0.92, and most of the paper's claim is gone.

The extension puts the trading cost inside the objective instead of subtracting it afterwards. A λ of 0.002
cuts turnover from 0.96 to 0.24 — a quarter of the trading — and gross Sharpe falls from 2.36 to 1.76.

And here is the honest result, the third number, in red. At 25 basis points the cost-aware model is *still
worse*: 1.52 against 1.64. It only wins at 50, where it gets 1.27 against 0.92. The crossover is about 31
basis points one-way, and you can see it on the chart.

So the answer is conditional. If you trade large-cap U.S. equities you should not bother. If you are paying
more than about thirty basis points you should. That is a more useful statement than the one we made last
term.

*If asked:* λ was one choice, not a sweep. A sweep costs three minutes per config on this GPU.

---

## 11 · Extension 2: stability and admissibility — 0:55

Two things, one slide.

**Left — our other 230P objection.** The paper reports one Sharpe for a twenty-five-year window. Here it is
year by year. The first five years average 6.15 annualised; the last five average 2.42. Two negative years,
2003 and 2009. 1995 alone is 13. So the model is roughly three times better in its first decade out of sample
than in its last, and the single number 2.6 hides that entirely. That is the argument for re-estimation, and
it is now evidence rather than an assertion.

**Right — the extension the paper skips.** A pricing kernel has to be positive or there is an arbitrage. The
paper checks this in sample and never imposes it. We added one line: penalise the squared negative part of M.

And we got a null result, which is worth reporting properly. The baseline never goes negative — zero months
out of 300, minimum M is 0.944 — so there was nothing to fix at the scale we measure M. Meanwhile the penalty
costs two thirds of the Sharpe, 0.68 down to 0.23.

The reason it is unmeasurable is our fault, not the paper's, and it is in the last bullet: we report M on
L1-normalised weights, which pins it near one, while the penalty acts on the raw weights during training —
and the raw weights are the ones we did not save. Two-line fix, on the list.

---

## 12 · Next — 0:25

*Four cards. Do not read them all.*

The one to say out loud is bottom right, because it is this course. The cost-aware SDF is a sequential
decision problem: state is characteristics plus macro state plus the book you are already holding, the action
is the portfolio weights, and the reward is pricing error minus trading cost. The turnover penalty we just
showed is the myopic one-step version of that. An actor–critic formulation is the natural next step, and it
is the same code.

The other three in one sentence: extend the sample past 2016 with WRDS, put confidence bands on everything
given the seed spread we showed, and stress the adversary.

---

## 13 · Questions

**Likely questions, short answers:**

* **Why is your Sharpe below theirs?** One fortieth of the compute, no hyper-parameter search, and our
  training Sharpe is *above* theirs — the fit ran slightly further before validation stopped it. The gap is
  about one and a half ensemble standard deviations.
* **Did you replicate the FFN, elastic net and OLS benchmarks on real data?** No. Those rows on slide 8 are
  the paper's and are labelled as such. We ran our own FFN and OLS only in the simulation.
* **Is 0.68 versus 0.75 significant?** We cannot say without the block bootstrap on the next-steps slide, and
  that is exactly the point of reporting the seed spread.
* **Why is your XS-R² 0.228 and not 0.045?** Two definitions. The authors' notebook prints the T_i-weighted
  one, and that is what their 0.23 is. Unweighted we get 0.045.
* **Did you change the model to get these numbers?** No. Both extensions are opt-in flags and both are off in
  the replication config. What we changed was fidelity to their TensorFlow graph — the appendix slide lists
  everything that still differs.
* **Why only four seeds on the raw-macro ablation?** Compute. The conclusion — collapse — matches the paper,
  but treat the 0.07 as fragile.

---

## 14 · Appendix — backup only

Do not present unless asked.

Framing: the eight things that were wrong in our first run were all fixed and are documented in the repo's
`notes.md`. This table is what still differs *on purpose*. The first row is the honest answer to "why is your
number lower" — we ran their optimum, not their search.

Row four is the one to volunteer before someone finds it: we did not re-run the FFN, elastic-net and OLS
benchmarks on real data.
