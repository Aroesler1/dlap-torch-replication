const pptxgen = require("pptxgenjs");
const fs = require("fs"), path = require("path");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5

// Real plot directory. Resolves whether this script sits next to the clone (./repo/output)
// or inside the repo itself (repo/deck/build_deck.js -> ../output).
const PLOTS = [path.join(__dirname, "repo", "output"), path.join(__dirname, "..", "output")].find(p => fs.existsSync(p));
if (!PLOTS) throw new Error("plots directory not found: expected ./repo/output or ../output");
const img = (rel) => path.join(PLOTS, rel);
// Deck charts built by make_charts.py from the same results.json. Native pptx charts only render
// in PowerPoint (Keynote / Google Slides / most PDF converters drop them), so these are images.
const CHARTS = [path.join(__dirname, "repo", "deck", "charts"), path.join(__dirname, "charts")].find(p => fs.existsSync(p));
if (!CHARTS) throw new Error("charts directory not found: run `python3 make_charts.py` in the deck folder");
const chart = (rel) => path.join(CHARTS, rel);

const NAVY = "1E2A47", GOLD = "B8860B", CARD = "EEF2F8", CREAM = "FDF6E8", GREY = "5A6474", WHITE = "FFFFFF", LIGHT = "D8DEE9", RED = "9E2A2B", GREEN = "2E6B4F";
const TITLE_FONT = "Cambria", BODY_FONT = "Calibri";
const FILL = { color: GOLD, bold: true }; // style for numbers still to be filled from the VM run
const nav = "Recap  ·  Replication  ·  Challenges  ·  Results  ·  Extensions  ·  Next";

function header(slide, kicker, title) {
  slide.background = { color: WHITE };
  slide.addText(kicker, { x: 0.6, y: 0.3, w: 8, h: 0.3, fontFace: BODY_FONT, fontSize: 11, bold: true, color: GOLD, margin: 0, isTextBox: true });
  slide.addText(title, { x: 0.6, y: 0.6, w: 12, h: 0.8, fontFace: TITLE_FONT, fontSize: 32, bold: true, color: NAVY, margin: 0, isTextBox: true });
  slide.addText(nav, { x: 0.6, y: 7.0, w: 12, h: 0.3, fontFace: BODY_FONT, fontSize: 9, color: "9AA3B2", margin: 0, isTextBox: true });
}
function card(slide, x, y, w, h, label, head, body, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: opts.fill || CARD }, line: { color: opts.fill || CARD } });
  let yy = y + 0.15;
  if (label) { slide.addText(label, { x: x + 0.2, y: yy, w: w - 0.4, h: 0.28, fontFace: BODY_FONT, fontSize: 10, bold: true, color: GOLD, margin: 0, isTextBox: true }); yy += 0.3; }
  if (head) { slide.addText(head, { x: x + 0.2, y: yy, w: w - 0.4, h: 0.4, fontFace: BODY_FONT, fontSize: 15, bold: true, color: NAVY, margin: 0, isTextBox: true }); yy += 0.45; }
  if (body) slide.addText(body, { x: x + 0.2, y: yy, w: w - 0.4, h: h - (yy - y) - 0.15, fontFace: BODY_FONT, fontSize: opts.fs || 12, color: GREY, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 4 });
}
function bullets(items, fs = 13) {
  return items.map((t, i) => (typeof t === "string" ? { text: t, options: { bullet: true, breakLine: i < items.length - 1, fontSize: fs, color: GREY, paraSpaceAfter: 6 } } : t));
}
function bigstat(slide, x, y, w, num, label, color = NAVY) {
  slide.addText(num, { x, y, w, h: 0.9, fontFace: TITLE_FONT, fontSize: 44, bold: true, color, margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 0.9, w, h: 0.7, fontFace: BODY_FONT, fontSize: 12, color: GREY, margin: 0, isTextBox: true, valign: "top" });
}
function tbl(slide, rows, x, y, w, colW, fs = 11) {
  const data = rows.map((r, i) => r.map((c, j) => {
    const o = { fontFace: BODY_FONT, fontSize: fs, color: i === 0 ? WHITE : NAVY, bold: i === 0 || j === 0, fill: { color: i === 0 ? NAVY : (i % 2 ? WHITE : CARD) }, align: j === 0 ? "left" : "center", valign: "middle", margin: [3, 6, 3, 6] };
    if (typeof c === "object") return { text: c.text, options: Object.assign(o, c.options || {}) };
    if (c === "[ ]") return { text: "[ ]", options: Object.assign(o, { color: GOLD, bold: true }) };
    return { text: c, options: o };
  }));
  slide.addTable(data, { x, y, w, colW, border: { type: "solid", pt: 0.5, color: LIGHT }, rowH: 0.32 });
}

// ---------------------------------------------------------------- 1 title
let s = pres.addSlide();
s.background = { color: NAVY };
s.addText("MFE 230ZA  ·  GROUP 14  ·  REPLICATION & EXTENSIONS", { x: 0.8, y: 1.6, w: 11, h: 0.4, fontFace: BODY_FONT, fontSize: 13, bold: true, color: GOLD, margin: 0, isTextBox: true });
s.addText("Deep Learning\nin Asset Pricing", { x: 0.8, y: 2.1, w: 11, h: 2.4, fontFace: TITLE_FONT, fontSize: 60, bold: true, color: WHITE, margin: 0, isTextBox: true });
s.addText("Chen, Pelger & Zhu (Management Science, 2024): replicating the adversarial SDF in PyTorch,\nwhat broke along the way, and what we built on top of it.", { x: 0.8, y: 4.6, w: 11.5, h: 1, fontFace: BODY_FONT, fontSize: 18, color: "CAD3E3", margin: 0, isTextBox: true });
s.addText("Aditya Aryan  ·  Giancarlo Alves  ·  Hashim Almodamagha  ·  Thomas Claudel  ·  Alex Roesler", { x: 0.8, y: 6.2, w: 11.5, h: 0.35, fontFace: BODY_FONT, fontSize: 13, color: "CAD3E3", margin: 0, isTextBox: true });
s.addText("September 11, 2026", { x: 0.8, y: 6.58, w: 11, h: 0.35, fontFace: BODY_FONT, fontSize: 12, color: "9AA3B2", margin: 0, isTextBox: true });
s.addNotes("[0:20]  Total budget 15 min including questions, so ~10:30 of slides.\n\nSay: Last term we wrote a referee report on Chen, Pelger and Zhu. This term we ran it. Everything you will see comes out of a PyTorch port of their TensorFlow code, on their own data, on an H100 — and the interesting part is not that it worked, it is the two places where it did not.\n\nPer-slide budget: 2) 0:55  3) 0:40  4) 0:50  5) 0:50  6) 1:15  7) 0:55  8) 1:40  9) 0:45  10) 1:05  11) 0:55  12) 0:25. Slide 14 is a backup for questions.");

// ---------------------------------------------------------------- 2 recap
s = pres.addSlide(); header(s, "RECAP / 230P IN ONE MINUTE", "Where we left off, and what is new today");
card(s, 0.6, 1.6, 6.0, 4.9, "LAST TERM: A REFEREE REPORT", "The claim, and our two objections", null);
s.addText(bullets([
  "The paper estimates the SDF for ~10,000 U.S. stocks with three networks: an LSTM for macro states, a feedforward net for the SDF weights ω, and an adversary that picks the hardest test assets. The loss is the no-arbitrage condition E[M R] = 0, not a forecast error.",
  "Headline: out-of-sample Sharpe 2.6 (annual), EV 8%, XS-R² 23%; beats FFN, elastic net and OLS.",
  "Our objection 1: fitted once in 1986, never re-estimated, reported as one number over 25 years.",
  "Our objection 2: turns over ≈100% of the book every month, every Sharpe ratio is gross of costs.",
], 12.5), { x: 0.8, y: 2.5, w: 5.6, h: 3.9, margin: 0, isTextBox: true, valign: "top" });
card(s, 6.9, 1.6, 5.85, 4.9, "TODAY", "From reading the paper to running it", null, { fill: CREAM });
s.addText(bullets([
  "Replicated the paper end-to-end: simulation (Table II) and U.S. equities (Table III) with the authors' own data.",
  "Ported the (dead) TensorFlow-1.12 code to PyTorch; every design choice is now inspectable and switchable.",
  "Ran the ablations that isolate why it works: no adversary, no macro, raw macro without LSTM.",
  "Built both 230P improvements into the code: transaction costs inside the objective, and a year-by-year test.",
  "Added a third extension the paper skips: enforcing an admissible (positive) SDF.",
], 12.5), { x: 7.1, y: 2.5, w: 5.5, h: 3.9, margin: 0, isTextBox: true, valign: "top" });

s.addNotes("[0:55]  One minute only — everybody has read the paper.\n\nLeft, fast: three networks, and the loss is the no-arbitrage condition, not a forecast error. Headline out-of-sample Sharpe 2.6 annual. Our two objections last term were: fitted once in 1986 and reported as a single number for 25 years, and it turns the whole book over every month with every Sharpe quoted gross.\n\nRight: what is new today. We replicated it end to end, ported the dead TensorFlow code, ran the ablations, and built both 230P objections into the objective — plus a third extension the paper skips.\n\nThen move on. Do not re-explain the model.");

// ---------------------------------------------------------------- 3 replication plan / pipeline
s = pres.addSlide(); header(s, "REPLICATION / PIPELINE", "What we replicated, and how the pieces fit");
const steps = [
  ["DATA", "Authors' npz panels", "Returns + 46 rank-normalised characteristics per stock-month; 178 macro series. Train 1967–86, valid 1987–91, test 1992–2016."],
  ["STAGE 1", "Unconditional", "SDF net on E[M R_i] = 0 for every stock (g = 1). 256 epochs, Adam, dropout 0.05."],
  ["STAGE 2", "Adversary", "Freeze ω; train g(I, LSTM₃₂) with 8 tanh outputs to maximise the pricing error. 64 steps."],
  ["STAGE 3", "Conditional", "Freeze g; retrain ω on the 8 adversarial moments. 1024 epochs. Keep best validation Sharpe."],
  ["ENSEMBLE", "9 seeds → 1 model", "Average ω across seeds, L1-normalise each month, build the SDF factor F = −ω'R."],
  ["METRICS", "SR · EV · XS-R²", "Second-stage β-network on R·F; project returns on β for EV and cross-sectional R²."],
];
steps.forEach((st, i) => {
  const x = 0.6 + i * 2.05;
  card(s, x, 1.7, 1.9, 3.3, st[0], st[1], st[2], { fs: 10.5 });
  if (i < steps.length - 1) s.addText("→", { x: x + 1.88, y: 3.1, w: 0.2, h: 0.4, fontSize: 16, color: GOLD, margin: 0, isTextBox: true, align: "center" });
});
card(s, 0.6, 5.3, 12.15, 1.4, "REPLICATED", null, "Table II (simulation, two DGPs) · Table III (U.S. equities: SR, EV, XS-R² for GAN vs. benchmarks) · Figure 6 ablations (no adversary, no macro, 178 raw macro series) · Figures 8, 9, 13 (variable importance, β-sorted deciles, LSTM macro states). Same architecture as the paper's Table I optimum: FFN 2×64, 4 SDF states, 32 adversary states, 8 moments, lr 0.001.", { fill: CREAM, fs: 12 });

s.addNotes("[0:40]  The pipeline, left to right. Point at the boxes, do not read them.\n\nThe two things worth saying out loud: stage 2 is where the adversary picks the hardest test assets, which is the whole idea of the paper; and everything is a 9-seed ensemble, which turns out to matter more than we expected — that comes back on slide 8.\n\nBottom strip: this is the full list of what we reproduced. Same architecture as their Table I optimum.");

// ---------------------------------------------------------------- 4 data
s = pres.addSlide(); header(s, "REPLICATION / DATA", "The data: what the authors share, and what they don't");
tbl(s, [
  ["Split", "Months", "Window", "Permnos in split", "Stock-months", "Avg stocks / month"],
  ["Train", "240", "1967–1986", "3,686", "336,113", "1,400.5"],
  ["Valid", "60", "1987–1991", "3,347", "132,167", "2,202.8"],
  ["Test", "300", "1992–2016", "7,141", "750,275", "2,500.9"],
], 0.6, 1.7, 7.4, [1.2, 1.0, 1.4, 1.4, 1.2, 1.2]);
s.addText("1,218,555 stock-months. Shapes and windows match the paper exactly.", { x: 0.6, y: 3.05, w: 7.4, h: 0.3, fontSize: 10, italic: true, color: GREY, margin: 0, isTextBox: true });
card(s, 0.6, 3.5, 7.4, 3.2, null, "Inputs, exactly as in the paper", null);
s.addText(bullets([
  "46 characteristics (past returns, investment, profitability, intangibles, value, frictions), each ranked cross-sectionally every month into [−0.5, 0.5]. A stock enters only when all 46 are available: ~31,000 CRSP names → ~10,000.",
  "178 macro series: 124 FRED-MD, the 46 characteristic medians, 8 Welch–Goyal predictors; McCracken–Ng transformations, then standardised with training-window moments only.",
  "Missing return = −99.99 sentinel → boolean mask; the loss weights each stock by its number of months T_i.",
], 11.5), { x: 0.8, y: 4.0, w: 7.0, h: 2.6, margin: 0, isTextBox: true, valign: "top" });
card(s, 8.3, 1.7, 4.45, 5.0, "WHAT IS NOT SHARED", "Three gaps we had to work around", null, { fill: CREAM });
s.addText(bullets([
  "No PERMNO identifiers. The stock index is consistent inside a split but not across splits, and each split holds only 3.7k / 3.3k / 7.1k names — the paper's \"~10,000\" is the union over the whole sample. Enough for turnover, not enough to join anything new.",
  "No raw CRSP/Compustat: the characteristic construction (Table A.VII) cannot be re-run, so 2017–2026 cannot be appended without WRDS.",
  "Macro is pre-transformed and final-vintage: our 230P real-time critique still stands and cannot be tested from this data.",
], 11.5), { x: 8.5, y: 2.65, w: 4.05, h: 3.9, margin: 0, isTextBox: true, valign: "top" });

s.addNotes("[0:50]  The authors share pre-built npz panels, and that is both the good news and the bad news.\n\nGood news: the shapes match the paper on the first load — 240 / 60 / 300 months, 46 characteristics, 178 macro series, 1.2 million stock-months. Nothing to reconstruct.\n\nBad news, right-hand card: no PERMNOs, so we cannot join anything new; no raw CRSP, so the characteristics cannot be rebuilt and the sample cannot be extended past 2016; and the macro data is final-vintage and pre-transformed, so our 230P real-time critique cannot even be tested from this file.\n\nOne surprise worth a sentence: each split holds three to seven thousand stocks, not ten thousand. The ten thousand is the union over the whole sample.");

// ---------------------------------------------------------------- 5 method as implemented
s = pres.addSlide(); header(s, "REPLICATION / METHOD", "The objective we actually optimise");
card(s, 0.6, 1.7, 6.0, 2.3, "PRICING LOSS (authors' code, line for line)", null, null);
s.addText([
  { text: "L(ω, g) = mean over j, i of  (T_i / T_max) · [ (1/T_i) Σ_t  M_{t+1} · R_{t+1,i} · g_{j,t,i} ]²", options: { fontSize: 14, color: NAVY, bold: true, breakLine: true } },
  { text: "M_{t+1} = 1 + Σ_i ω(I_{t,i}, h_t) R_{t+1,i}       h_t = LSTM₄(macro_{1..t})       g_{j} = tanh(FFN(I, LSTM₃₂(macro)))", options: { fontSize: 12, color: GREY, breakLine: true } },
  { text: "Stage 1 sets g ≡ 1. Stage 2 maximises L in g. Stage 3 minimises L in ω with g frozen.", options: { fontSize: 12, color: GREY } },
], { x: 0.8, y: 2.15, w: 5.6, h: 1.7, margin: 0, isTextBox: true, valign: "top" });
card(s, 0.6, 4.2, 6.0, 2.5, "SELECTION & ENSEMBLE", null, null);
s.addText(bullets([
  "Every epoch: unconditional loss and Sharpe on train / valid / test. Checkpoints kept for best validation loss (used to move between stages) and best validation Sharpe (used for the ensemble).",
  "Nothing in the 1992–2016 window is used for any decision; it is printed for the training-curve plot only.",
  "Ensemble = mean of raw ω over 9 seeds, then L1-normalised per month, so F is the return on $1 gross.",
], 11.5), { x: 0.8, y: 4.65, w: 5.6, h: 2.0, margin: 0, isTextBox: true, valign: "top" });
s.addImage({ path: img("gan/plots/training_curves.png"), x: 6.9, y: 1.72, w: 5.85, h: 1.91 });   // 1650x540 → aspect 3.056
s.addText("Seed 0. At the dashed line the SDF net is reset to its best-validation-loss weights and the adversarial moments take over: training Sharpe then runs away to 3.0 while validation and test stay flat at 0.7 — the ensemble is what keeps that gap honest.", { x: 6.9, y: 3.72, w: 5.85, h: 1.15, fontSize: 10.5, color: GREY, italic: true, margin: 0, isTextBox: true });
tbl(s, [["Hyperparameter", "Paper (Table I)", "Ours"], ["SDF net layers × units", "2 × 64", "2 × 64"], ["SDF / adversary LSTM states", "4 / 32", "4 / 32"], ["Adversary layers / moments", "0 / 8", "0 / 8"], ["lr · dropout · epochs", "0.001 · 0.95 · 256/64/1024", "same"]], 6.9, 5.00, 5.85, [1.9, 2.35, 1.6], 9.5);

s.addNotes("[0:50]  The objective, in one line. This is the only equation in the deck.\n\nRead it as: for every test asset j and every stock i, the average of M times the excess return times the adversarial weight should be zero. Stage 1 sets g to one — that is the unconditional model. Stage 2 maximises this in g. Stage 3 minimises it in omega.\n\nSelection card: nothing in 1992–2016 touches any decision. Test Sharpe is printed every epoch only so we can draw the picture on the right — and that picture is the honest story of the method: training Sharpe goes to 3.0, validation and test sit at 0.7.\n\nTable: our hyper-parameters are theirs.");

// ---------------------------------------------------------------- 6 challenges
s = pres.addSlide(); header(s, "CHALLENGES", "Four things that did not just work");
const ch = [
  ["1 · THE CODE IS DEAD", "TensorFlow 1.12, Python 3.6, 2019 APIs", "placeholders, dynamic_rnn, boolean_mask, tf.train.Saver: none of it installs on a 2026 machine. Getting the GPU was its own detour — pip resolved the CPU wheel until we installed torch from the cu128 index alone.",
   "Rewrote the model in PyTorch against the TF graph, function by function: same data layer, same masked loss with the T_i weighting, same three-stage schedule and checkpoint rules. Verified on a synthetic panel with a planted SDF, then on the paper's own simulation DGP, before touching CRSP data."],
  ["2 · DEFAULTS ARE PART OF THE MODEL", "The first full run had the shape but not the numbers", "Test Sharpe 0.63 against the paper's 0.75, and the cost-aware extension was nearly worthless at 0.09. Nothing was obviously broken. Framework defaults were silently doing the work: initialisers, where Adam's epsilon sits, whether dropout is on for a frozen network.",
   "Line-by-line audit against their graph and notebook, then eight fixes: TF Glorot init with LSTM forget bias 1, a tf.train.Adam-exact update with a fresh optimiser per stage, dropout left active on the frozen nets, the adversary's 4×64 loop, ignore_epoch 32. Headline 0.63 → 0.68; the cost extension 0.09 → 0.51."],
  ["3 · THE SEARCH IS TOO BIG", "384 configs × validation × 4 finalists × 9 seeds", "Even on an H100 a full re-tune is days, not an evening. And selection on validation Sharpe means the paper's test numbers are themselves conditional on that search.",
   "Fixed the architecture at their reported optimum (Table I) and spent the compute on 9 seeds per model plus the ablations — about one fortieth of their budget. Then report the seed dispersion, which the paper does not."],
  ["4 · STATE THAT CARRIES OVER", "LSTM memory across train → valid → test", "The macro state at January 1992 depends on 25 years of history. Evaluate the test window from a zero state and the model prices with amnesia.",
   "Run the LSTM over the concatenated macro history and slice the window — what the authors' getNextInitialState does. The same trick lets us read the four hidden states over the whole sample."],
];
ch.forEach((c, i) => {
  const x = 0.6 + i * 3.05;
  card(s, x, 1.62, 2.95, 5.15, c[0], null, null);
  s.addText(c[1], { x: x + 0.2, y: 2.05, w: 2.55, h: 0.55, fontSize: 11.5, bold: true, color: NAVY, margin: 0, isTextBox: true, valign: "top" });
  s.addText(c[2], { x: x + 0.2, y: 2.68, w: 2.55, h: 1.6, fontSize: 10, color: GREY, margin: 0, isTextBox: true, valign: "top" });
  s.addText("HOW WE DEALT WITH IT", { x: x + 0.2, y: 4.32, w: 2.55, h: 0.24, fontSize: 9, bold: true, color: GOLD, margin: 0, isTextBox: true });
  s.addText(c[3], { x: x + 0.2, y: 4.58, w: 2.55, h: 2.1, fontSize: 10, color: NAVY, margin: 0, isTextBox: true, valign: "top" });
});
s.addNotes("[1:15]  This is the slide the assignment is really about, so slow down here.\n\n1. The code is dead. TensorFlow 1.12 on Python 3.6 does not install in 2026. And before any of that, a plain pip install gave us the CPU build of torch — the H100 sat idle until we reinstalled from the CUDA index alone.\n\n2. This is the one worth telling. Our first complete run looked right — right shapes, right pictures, ablations in the right order — and was wrong. Test Sharpe 0.63, and our cost-aware extension came out at 0.09, basically useless. Nothing was broken. PyTorch just has different defaults from TensorFlow: different initialisers, epsilon in a different place inside Adam, and dropout switched off on a frozen network where TensorFlow feeds keep-prob to the whole graph. Eight fixes later the headline went to 0.68 and the extension went to 0.51 — a five-fold move from changes that are invisible in a results table. If you take one thing from the replication, take that.\n\n3. We could not redo their 384-configuration search, so we took their optimum and spent the compute on seeds instead.\n\n4. The LSTM state has to carry across the split boundary or the model prices 1992 with amnesia.");

// ---------------------------------------------------------------- 7 result: simulation
s = pres.addSlide(); header(s, "RESULTS / 1 · SIMULATION (TABLE II)", "On the paper's own DGP, the port recovers the SDF");
s.addText("R = β F + ε,  F ~ N(0.32, 0.1),  ε ~ N(0,1),  N = 500,  T = 250 / 100 / 250.  Setup 1: β = C₁·C₂ (a pure interaction). Setup 2: β = C·sign(h_t), h_t a noisy sine cycle observed only through its trended increment.", { x: 0.6, y: 1.55, w: 12.1, h: 0.6, fontSize: 12, color: GREY, margin: 0, isTextBox: true });
tbl(s, [
  ["Setup 1: two characteristics", "SR train", "SR valid", "SR test", "EV test", "XS-R² test"],
  ["Population SDF (paper)", "0.96", "1.09", "0.94", "0.17", "0.17"],
  ["GAN (paper)", "0.98", "1.11", "0.94", "0.13", "0.07"],
  ["GAN (ours, 3 seeds)", {text:"0.99",options:{bold:true}}, {text:"0.92",options:{bold:true}}, {text:"1.00",options:{bold:true}}, {text:"0.17",options:{bold:true}}, {text:"0.12",options:{bold:true}}],
  ["FFN forecast (paper / ours)", "0.94 / 0.97", "1.04 / 0.92", "0.89 / 1.02", "0.05 / 0.17", "−0.33 / 0.13"],
  ["Linear (paper / ours)", "0.07 / 0.12", "−0.10 / 0.04", "0.01 / −0.09", "0.00 / 0.00", "0.01 / 0.00"],
], 0.6, 2.25, 7.3, [2.6, 0.94, 0.94, 0.94, 0.94, 0.94], 10.5);
tbl(s, [
  ["Setup 2: char + hidden macro state", "SR test", "EV test", "XS-R² test"],
  ["Population (paper)", "0.86", "0.17", "0.15"],
  ["GAN with LSTM (paper / ours)", "0.64 / 0.60", "0.17 / 0.01", "0.15 / 0.00"],
  ["FFN, last macro obs (paper / ours)", "0.06 / 0.02", "0.02 / 0.16", "0.02 / 0.20"],
], 0.6, 4.45, 7.3, [3.1, 1.4, 1.4, 1.4], 10.5);
card(s, 8.2, 2.25, 4.55, 1.95, "WHAT THIS BUYS US", null, null, { fill: CREAM });
s.addText(bullets([
  "Setup 1 is the clean check on the port: GAN test Sharpe 1.00 against a population SDF of 1.02 and the paper's 0.94, while the linear model finds nothing in the interaction — exactly the paper's result.",
  "Costs minutes on a GPU, so any change to the objective can be regression-tested here before it touches CRSP.",
], 10.5), { x: 8.4, y: 2.66, w: 4.15, h: 1.45, margin: 0, isTextBox: true, valign: "top" });
s.addText("SETUP 2 · THE LSTM RECOVERS A CYCLE IT NEVER SEES", { x: 8.2, y: 4.32, w: 4.55, h: 0.24, fontSize: 9.5, bold: true, color: GOLD, margin: 0, isTextBox: true });
s.addImage({ path: img("sim2/hidden_state_recovery.png"), x: 8.2, y: 4.58, w: 4.4, h: 2.2 });   // 1500x750 → aspect 2.0
s.addText("Setup 2 caveat: the loading is C·sign(h_t), and our β-network sees characteristics only — as the authors' own config does — so no function of C can represent it and EV / XS-R² are ≈0 by construction, not by failure. The SDF weights, which do see the state, still reach 0.60 against the paper's 0.64 — though our population SDF scores 0.97 there against their 0.86, so the two calibrations are not identical. Our FFN and LS benchmarks are our own code (the authors released no simulation script) and are stronger than theirs.", { x: 0.6, y: 5.85, w: 7.3, h: 0.95, fontSize: 9.5, italic: true, color: GREY, margin: 0, isTextBox: true });
s.addNotes("[0:55]  Before the real data: does the port work at all? The simulation is where you can answer that, because there is a population SDF to compare against.\n\nSetup 1, top table: our GAN gets test Sharpe 1.00, the population SDF gets 1.02, the paper gets 0.94. The linear model gets minus 0.09 — it cannot see an interaction between two characteristics, which is the point the paper is making. So the port is right.\n\nSetup 2, bottom: the loading flips sign with a hidden macro cycle that is only observable through a noisy trended increment. Our Sharpe 0.60 against the paper's 0.64. The picture on the right is the one to point at: top panel is the true hidden state, bottom is what the LSTM reconstructed from the increment. Correlations 0.45 to 0.64 across the four units. That is the mechanism the whole paper rests on.\n\nIf asked about the two zeros in the bottom table: our β-network is fitted on characteristics only, exactly as their released config does it, and the true loading depends on the macro state — so those two metrics are zero by construction. The Sharpe, which uses the weights, is fine.");

// ---------------------------------------------------------------- 8 result: real data headline
s = pres.addSlide(); header(s, "RESULTS / 2 · U.S. EQUITIES (TABLE III)", "The ordering replicates; the level lands 9 % low");
tbl(s, [
  ["Model", "SR train", "SR valid", "SR test (monthly)", "SR test (annual)", "EV test", "XS-R² test"],
  ["GAN, hidden macro states  (paper)", "2.68", "1.43", "0.75", "2.60", "0.08", "0.23"],
  ["GAN, hidden macro states  (ours, 9 seeds)", {text:"2.99",options:{bold:true}}, {text:"1.36",options:{bold:true}}, {text:"0.68",options:{bold:true}}, {text:"2.36",options:{bold:true}}, {text:"0.084",options:{bold:true}}, {text:"0.228",options:{bold:true}}],
  ["   spread across the 9 seeds (test SR)", "", "", {text:"0.283 – 0.750",options:{color:GOLD,bold:true}}, "", "", ""],
  ["UNC: no adversary  (paper / ours)", "1.93 / 1.52", "1.33 / 0.72", "0.53 / 0.45", "1.84 / 1.57", "0.07 / 0.104", "0.19 / 0.287"],
  ["GAN, no macro  (paper / ours)", "1.90 / 1.75", "1.35 / 1.49", "0.69 / 0.62", "2.39 / 2.15", "— / 0.101", "— / 0.258"],
  ["GAN, 178 raw macro, no LSTM  (paper / ours)", "1.07 / 1.03", "0.05 / 0.02", "0.05 / 0.07", "0.17 / 0.23", "— / 0.117", "— / 0.279"],
  ["FFN forecast  (paper)", "0.45", "0.42", "0.44", "1.52", "0.04", "0.15"],
  ["Elastic net  (paper)", "1.37", "1.15", "0.50", "1.73", "0.04", "0.19"],
], 0.6, 1.65, 12.15, [4.0, 1.3, 1.3, 1.55, 1.5, 1.2, 1.3], 10.5);
card(s, 0.6, 4.9, 3.9, 1.7, "THE HEADLINE: 0.68 VS 0.75", null, "9 % low, on one fortieth of the compute and with no hyper-parameter search. Train came in above the paper (2.99 vs 2.68), so our fit ran slightly further before validation stopped it. EV and XS-R² land on the paper: 0.084 / 0.228 vs 0.08 / 0.23.", { fill: CREAM, fs: 10.5 });
card(s, 4.7, 4.9, 3.9, 1.7, "THE ORDERING REPLICATES — BUT NOT ON EV", null, "Sharpe ranks exactly as in the paper: adversary 0.68 > none 0.45, and states 0.68 > no macro 0.62 > raw macro 0.07. EV and XS-R² do not rank at all: every ablation, including the run that collapses to 0.07, scores as high or higher.", { fill: CREAM, fs: 10.5 });
card(s, 8.8, 4.9, 3.95, 1.7, "WHAT THE PAPER NEVER REPORTS", null, "Across 9 seeds the test Sharpe runs 0.28 to 0.75 — a spread wider than the paper's gap between the GAN and OLS. The ensemble (0.68) beats the average seed (0.49) and 8 of the 9, but not seed 0.", { fill: CREAM, fs: 10.5 });
s.addText("Ours = 9-seed ensembles of best-validation-Sharpe checkpoints; the raw-macro ablation has 4 seeds. XS-R² is the T_i-weighted definition the authors' notebook prints — unweighted it is 0.045.", { x: 0.6, y: 6.68, w: 12.15, h: 0.25, fontSize: 8.5, italic: true, color: GREY, margin: 0, isTextBox: true });
s.addNotes("[1:40]  The main slide. Give it time.\n\nRow 1 is the paper, row 2 is us: 0.68 against 0.75 monthly, 2.36 against 2.60 annual. We are 9 % low. The most likely reason is right there in row 2 — our training Sharpe is 2.99 against their 2.68, so our fit ran a bit further into the training window before validation stopped it, and we did none of their 384-configuration search. EV 0.084 and XS-R² 0.228 sit on their 0.08 and 0.23.\n\nThen the ablations, and this is the part that replicates cleanly. Take the adversary away and it falls to 0.45. Take the macro states away and it falls to 0.62. Feed all 178 raw macro series without the LSTM and it collapses to 0.07 out of sample — the paper says the same thing. So the ordering is theirs.\n\nMiddle card: the honest finding. The ordering only replicates on Sharpe. On EV and cross-sectional R² the ablations all score as high as the headline model or higher — the run that collapses to 0.07 has the best EV in the table. Our second-stage β-network is fitted on characteristics alone, so it explains a similar amount whatever SDF factor you point it at. We would not present EV as evidence that this model is better than the alternatives.\n\nRight card, the number we would put in a referee report: across nine seeds the test Sharpe runs 0.28 to 0.75. The paper's own gap between the GAN and OLS is smaller than the spread between two seeds of the same model.");

// ---------------------------------------------------------------- 9 result: figures
s = pres.addSlide(); header(s, "RESULTS / 3 · WHAT THE MODEL LEARNED", "Two of the paper's three pictures come back");
function figtitle(x, y, t) { s.addText(t, { x, y, w: 6.0, h: 0.24, fontSize: 10.5, bold: true, color: NAVY, margin: 0, isTextBox: true }); }
figtitle(0.6, 1.53, "① Cumulative SDF factor — 9-seed ensemble, L1-normalised weights");
s.addImage({ path: img("gan/plots/cumulative_sdf_factor.png"), x: 0.6, y: 1.80, w: 6.0, h: 2.40 });    // 1500x600
figtitle(6.75, 1.53, "② Variable importance — mean |∂ω/∂char| on test, 9-model average");
s.addImage({ path: img("gan/plots/variable_importance_test.png"), x: 6.75, y: 1.80, w: 6.0, h: 2.40 }); // 1500x600
figtitle(0.6, 4.33, "③ The four LSTM macro states over the test window");
s.addImage({ path: img("gan/plots/macro_states_test.png"), x: 0.6, y: 4.60, w: 6.0, h: 2.10 });         // 1500x525
card(s, 6.75, 4.33, 6.0, 2.37, "HOW TO READ THESE", null, null, { fill: CREAM });
s.addText(bullets([
  "① Still climbing after 1992, and the drawdown is tiny: worst month −4.5 %, max drawdown 5.5 % over 25 years. This is the picture that makes the paper's claim look strong.",
  "② The shape, not the names: short-term reversal (0.085) and SUV (0.056) stand out, then 43 of 46 characteristics sit flat between 0.050 and 0.038. The paper's Fig. 8 is far more concentrated — we do not replicate which characteristics matter.",
  "③ The honest one. Paper Fig. 13 has four smooth cyclical states peaking in recessions the model never saw. Ours are high-frequency and show no visible business-cycle structure. The states are never written to disk, so we cannot quantify it — we can only report it.",
], 10), { x: 6.95, y: 4.72, w: 5.6, h: 1.9, margin: 0, isTextBox: true, valign: "top" });
s.addNotes("[0:45]  Three pictures, thirty seconds each.\n\nOne: the cumulative SDF factor. The two dashed lines are the split boundaries. It keeps climbing after 1992 and the worst month in 25 years is minus 4.5 % with a 5.5 % max drawdown. That is the picture that sells the paper.\n\nTwo: variable importance. Short-term reversal and SUV separate; then forty-three characteristics sit in a flat band. The paper's version is much more concentrated. So we reproduce the model's performance without reproducing its story about which characteristics matter. Do not over-claim here.\n\nThree, and this is the one that does not replicate. Their Figure 13 shows four smooth states that peak in recessions the network never saw — that is the headline picture of the macro half of the paper. Ours are noise at monthly frequency. We cannot even quantify it, because the state series are plotted and then thrown away rather than saved. That is on our list to fix.");

// ---------------------------------------------------------------- 10 extension 1 costs
s = pres.addSlide(); header(s, "EXTENSIONS / 1 · TRANSACTION COSTS", "The 2.36 survives 25 bps, not 50");
bigstat(s, 0.6, 1.7, 2.45, "0.96", "monthly turnover of the L1-normalised book on test — we guessed ≈1.0 in 230P");
bigstat(s, 3.3, 1.7, 2.45, "1.64", "annual test Sharpe net of 25 bps one-way costs (2.36 gross)");
bigstat(s, 6.0, 1.7, 2.45, "1.52", "same for the cost-aware model — at 25 bps the penalty still loses", RED);
s.addImage({ path: chart("net_of_cost_sharpe.png"), x: 8.8, y: 1.62, w: 3.95, h: 2.05 });   // built at exactly 3.95 x 2.05 in
card(s, 0.6, 3.85, 6.0, 2.2, "POST HOC (what we proposed in 230P)", null, "Each month subtract cost × turnover from the SDF-factor return and recompute the Sharpe at 0 / 10 / 25 / 50 bps. The portfolio never saw the cost, so this is a floor, not an estimate. Computed for every run in results.json. The paper's headline 2.6 is the 0 bps column and nothing else.", { fs: 11.5 });
card(s, 6.8, 3.85, 5.95, 2.2, "INSIDE THE LOSS (new)", null, "Add λ · E_t Σ_i |ω̃_{t,i} − ω̃_{t−1,i}| to the pricing loss on L1-normalised weights, λ = 0.002. Turnover 0.96 → 0.24 and gross Sharpe 2.36 → 1.76: the network pays four fifths of its trading to keep three quarters of its Sharpe. Net of cost it loses at 25 bps (1.52 vs 1.64) and wins at 50 (1.27 vs 0.92). Crossover ≈ 31 bps one-way.", { fill: CREAM, fs: 11.5 });
s.addNotes("[1:05]  Our first 230P objection, now inside the model.\n\nThe number on the left is the one we were guessing at last term: monthly turnover of 0.96. We estimated about 1.0 from their figures. So the criticism was right — this book turns over essentially in full every month.\n\nHow bad is it? Middle number: at 25 basis points one-way the headline 2.36 becomes 1.64. It survives. At 50 basis points it is 0.92, so most of the paper's claim is gone.\n\nThe extension: put the trading cost inside the objective instead of subtracting it afterwards. λ of 0.002 cuts turnover from 0.96 to 0.24 — a quarter of the trading — and gross Sharpe falls from 2.36 to 1.76.\n\nAnd here is the honest result, the third number on the left, in red: at 25 basis points the cost-aware model is still worse, 1.52 against 1.64. It only wins at 50, where it gets 1.27 against 0.92. The crossover is about 31 basis points one-way. So the answer is conditional: if you trade large-cap US equities you should not bother, and if you are paying more than about 30 basis points you should. That is a more useful statement than the one we made last term.\n\nCaveat if asked: λ was one choice, not a sweep. A sweep is three minutes per config on this GPU.");

// ---------------------------------------------------------------- 11 extension 2 stability + positivity
s = pres.addSlide(); header(s, "EXTENSIONS / 2 · STABILITY AND ADMISSIBILITY", "Two things the paper reports as one number, or not at all");
s.addImage({ path: chart("sharpe_by_year.png"), x: 0.6, y: 1.6, w: 6.3, h: 3.0 });   // built at exactly 6.3 x 3.0 in
card(s, 0.6, 4.75, 6.3, 1.95, "OUR 230P IMPROVEMENT 1, DELIVERED", null, "First five test years average 6.15 annualised, last five 2.42; 1992–2004 averages 4.72 against 2.43 for 2005–2016, with two negative years (2003, 2009). The single number 2.60 hides a model that is three times better in its first decade than its last. The expanding-window refit needs PERMNOs and stays on the 230ZB list.", { fs: 11 });
card(s, 7.1, 1.6, 5.65, 5.1, "ADMISSIBLE SDF (NEW)", "A pricing kernel must be positive", null, { fill: CREAM });
s.addText(bullets([
  "No-arbitrage requires M_{t+1} > 0. The paper checks this in-sample and never imposes it; nothing stops the estimated M from going negative in a crash month out of sample.",
  "We add λ · E[max(0, −M_{t+1})²] to the loss (one line). λ = 1.",
  "Result: M < 0 in 0 of 300 test months for the baseline already (minimum M = 0.944), and 0 of 300 with the penalty (minimum 0.946). The constraint never binds.",
  "It is not free: test Sharpe falls 0.68 → 0.23 and turnover 0.96 → 0.79 for no measurable gain in admissibility.",
  "Why it is unmeasurable: we report M on L1-normalised weights, so M sits in [0.94, 1.05] by construction, while the penalty acts on the raw weights during training — and only the normalised weights are saved. Fixing that is a two-line change we did not make before the deadline.",
], 11), { x: 7.3, y: 2.5, w: 5.25, h: 4.1, margin: 0, isTextBox: true, valign: "top" });
s.addNotes("[0:55]  Two things, one slide.\n\nLeft, our other 230P objection: the paper reports one Sharpe for a 25-year window. Here it is year by year. First five years average 6.15 annualised, last five 2.42. Two negative years, 2003 and 2009. Nineteen ninety-five alone is 13. So the model is roughly three times better in its first decade out of sample than in its last, and the single number 2.6 hides that entirely. That is the argument for re-estimation, and it is now evidence rather than an assertion.\n\nRight, the extension the paper skips. A pricing kernel has to be positive or there is an arbitrage; the paper checks this in sample and never imposes it. We added one line: penalise the squared negative part of M.\n\nAnd we got a null result that is worth reporting properly. The baseline never goes negative — zero months out of 300, minimum M is 0.944 — so there was nothing to fix at the scale we measure M. Meanwhile the penalty costs two thirds of the Sharpe, 0.68 down to 0.23.\n\nThe reason it is unmeasurable is in the last bullet, and it is our fault, not the paper's: we report M on normalised weights, which pins it near one, while the penalty acts on the raw weights during training — and the raw weights are the ones we did not save. Two-line fix, on the list.");

// ---------------------------------------------------------------- 12 next
s = pres.addSlide(); header(s, "NEXT / TOWARDS 230ZB AND A PAPER", "What we would do with WRDS access and another term");
const nx = [
  ["EXTEND THE SAMPLE", "Rebuild the 46 characteristics from CRSP/Compustat, append 2017–2026, and test the 1986 model through COVID and the 2022 rate cycle. Then the expanding-window refit from 230P."],
  ["INFERENCE, AND SAVE MORE STATE", "Block-bootstrap the test months across the 9-seed ensemble to put confidence bands on SR, EV and XS-R² differences — with a seed spread of 0.28 to 0.75, no comparison in this literature should be made without them. Also dump the raw weights and the LSTM states, the two things we could not check today."],
  ["STRESS THE ADVERSARY", "Vary the number of moments (4 / 8 / 16), run more than one adversarial round, and hold out characteristics from the adversary. Cheap with the current code."],
  ["THE RL LINK (THIS COURSE)", "The cost-aware SDF is a sequential decision: state = (characteristics, macro state, current book), action = ω, reward = pricing error + trading cost. A natural actor–critic formulation of the same problem."],
];
nx.forEach((n, i) => card(s, 0.6 + (i % 2) * 6.15, 1.65 + Math.floor(i / 2) * 2.55, 6.0, 2.4, n[0], null, n[1], { fs: 12, fill: i === 3 ? CREAM : CARD }));
s.addNotes("[0:25]  Four lines, do not read them all.\n\nThe one to say out loud is bottom right, because it is this course: the cost-aware SDF is a sequential decision problem. State is characteristics plus macro state plus the book you are already holding, action is the portfolio weights, reward is pricing error minus trading cost. The turnover penalty we showed is the myopic one-step version of that. An actor-critic formulation is the natural next step and it is the same code.\n\nThe other three in one sentence: extend the sample past 2016 with WRDS, put confidence bands on everything given the seed spread we showed, and stress the adversary.");

// ---------------------------------------------------------------- 13 questions
s = pres.addSlide(); s.background = { color: NAVY };
s.addText("Questions?", { x: 0.8, y: 2.6, w: 11, h: 1.4, fontFace: TITLE_FONT, fontSize: 60, bold: true, color: WHITE, margin: 0, isTextBox: true });
s.addText("Code, configs and every number in this deck: github.com/Aroesler1/dlap-torch-replication — run.py, simulate.py, summarize.py, deck/review.md", { x: 0.8, y: 4.2, w: 11.5, h: 0.6, fontFace: BODY_FONT, fontSize: 15, color: "CAD3E3", margin: 0, isTextBox: true });
s.addText("Backup slide follows: every deliberate deviation from the authors' implementation.", { x: 0.8, y: 4.85, w: 11.5, h: 0.4, fontFace: BODY_FONT, fontSize: 12, color: "9AA3B2", margin: 0, isTextBox: true });
s.addNotes("Likely questions and the short answers:\n\n• Why is your Sharpe below theirs? One fortieth of the compute, no hyper-parameter search, and our training Sharpe is above theirs — the fit ran slightly further before validation stopped it. The gap is about one and a half ensemble standard deviations.\n\n• Did you replicate the FFN / elastic net / OLS benchmarks on real data? No. Those rows on slide 8 are the paper's. We ran our own FFN and OLS only in the simulation.\n\n• Is 0.68 vs 0.75 significant? We cannot say without the bootstrap on the next-steps slide. That is exactly the point of the seed-spread number.\n\n• Why is your XS-R² 0.228 and not 0.045? Two definitions. The authors' notebook prints the T_i-weighted one and that is what their 0.23 is. Unweighted we get 0.045.\n\n• Did you change the model to get these numbers? No. The extensions are opt-in flags and both are off in the replication config. What we did change was fidelity to their TensorFlow graph — the appendix slide lists all eight.");

// ---------------------------------------------------------------- 14 appendix: deviations
s = pres.addSlide(); header(s, "APPENDIX", "Deviations from the authors' implementation (for the record)");
s.addText("Eight fidelity gaps were found by a line-by-line audit against the authors' graph and notebook, and all eight were fixed before the numbers in this deck were produced (test Sharpe 0.63 → 0.68, cost-aware extension 0.09 → 0.51). What remains below is what still differs, on purpose.", { x: 0.6, y: 1.55, w: 12.15, h: 0.45, fontSize: 11, color: GREY, margin: 0, isTextBox: true });
tbl(s, [
  ["Item", "Authors (TF 1.12)", "Ours (PyTorch)", "Effect"],
  ["Hyper-parameters", "384-config grid, best 4 on validation, 9 seeds each", "Their Table I optimum, 9 seeds, no search", "Same architecture, ~1/40 of the compute. Most likely source of 0.68 vs 0.75"],
  ["Random draws", "TF RNG, float32 reduction order", "PyTorch RNG, float32 reduction order", "Moves seed-level paths only; our seed spread is 0.283–0.750"],
  ["Simulation benchmarks", "No code released", "Our own FFN and LS implementations", "Ours are stronger than the paper's reported rows (setup 1 FFN EV 0.17 vs 0.05)"],
  ["Empirical benchmarks", "FFN, elastic net, OLS on CRSP", "Not run — the paper's values are quoted", "Slide 8 rows marked (paper)"],
  ["EV / XS-R²", "β network, T_i-weighted (notebook prints WXSR2)", "Same, plus the unweighted value and an ω-projection", "Extra diagnostic; weighted 0.228 vs unweighted 0.045"],
  ["Extensions", "—", "turnover_penalty, sdf_hinge, yearly Sharpe, net-of-cost Sharpe", "Opt-in flags, both off in configs/gan.json; the replication is pure"],
], 0.6, 2.1, 12.15, [2.2, 3.5, 3.3, 3.15], 9.5);
s.addNotes("Backup slide — do not present unless asked.\n\nThe framing to use: the eight things that were wrong in our first run were all fixed and are listed in the repo's notes.md. This table is what still differs on purpose. The first row is the honest answer to \"why is your number lower\": we ran their optimum, not their search.\n\nThe fourth row is the one to volunteer before someone finds it: we did not re-run the FFN, elastic-net and OLS benchmarks on real data. Those numbers on slide 8 are quoted from the paper and are labelled as such.");

pres.writeFile({ fileName: "230ZA_Presentation_Deck.pptx" }).then(() => console.log("written"));
