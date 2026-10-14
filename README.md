# Which tensors break a compressed chaotic surrogate?
### Matched-control and tensor-isolated perturbation experiments on Lorenz-63, Rössler and Lorenz-96 emulators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23157081.svg)](https://doi.org/10.5281/zenodo.23157081)

Author: Viren Singh ([ORCID 0009-0001-6618-0126](https://orcid.org/0009-0001-6618-0126))

Code, raw results and manuscript source for a controlled study of how pruning, quantization and random
parameter noise affect learned surrogates of chaotic flows. Every structured perturbation is paired with
random perturbations of exactly the same total relative norm, the same per-tensor norms and (for pruning) the
same support. On top of that, noise is injected **into one tensor at a time**, a first-order sensitivity
theory is derived and tested, and the conclusions are checked across network widths, a six-dimensional
system and mixed-precision quantization. **108,332 perturbed models** in total.

Pure NumPy / SciPy / pandas / Matplotlib / scikit-learn. No GPU and no deep-learning framework.

---

## Contents
1. [Main findings](#main-findings)
2. [Install and quick checks](#install-and-quick-checks)
3. [Reproduce the results](#reproduce-the-results)
   - [Option A: analysis only (minutes)](#option-a-analysis-only-minutes)
   - [Option B: recompute everything (hours)](#option-b-recompute-everything-hours)
4. [Which script produces which part of the paper](#which-script-produces-which-part-of-the-paper)
5. [Repository layout](#repository-layout)
6. [Result file formats](#result-file-formats)
7. [Reproducibility notes](#reproducibility-notes)
8. [Troubleshooting](#troubleshooting)
9. [License and citation](#license-and-citation)

---

## Main findings

Surrogates are residual MLPs `f(x) = x + W3 tanh(W2 tanh(W1 x + b1) + b2) + b3` (two hidden layers of 64
units unless stated), trained on one-step prediction and then perturbed after training. 30 seeds per system.

* **Failure dominates and the attractor survives only small displacements.** 68% (Lorenz-63), 79% (Rössler)
  and 49% (Lorenz-96) of perturbed surrogates diverge or collapse. Half of isotropic perturbations destroy the
  attractor at a relative parameter displacement of **4.4% (Lorenz-63), 1.8% (Rössler), 18% (Lorenz-96)**.
  Failure is rarely silent: at least 98% of failed models already have 10x their parent's one-step error.
* **Equal total norm is not equal damage.** At a total relative norm of 0.01, noise confined to the output
  layer `W3` fails 61% (Lorenz-63) / 76% (Rössler) of surrogates, isotropic noise 14% / 32%, and noise
  confined to the large middle matrix `W2` 0% / 9%. Removing `W3` from isotropic noise (it carries 4% of the
  squared norm) cuts failures by 47-74%.
* **Per-tensor relative displacement is the natural budget.** Measured against each tensor's own norm, the
  weight matrices of Lorenz-63 fail at similar radii (about 0.10-0.14). Across hidden widths 32-128 the
  weight-matrix radii stay in 0.06-0.24 but are **not** width-invariant.
* **A first-order sensitivity theory predicts the radii from the parent alone.** The one-step error caused by
  noise in a tensor is `dMSE ~ (|delta|^2 / n) * sum_k s_k` with analytic sensitivities `s_k`. Predicted vs
  measured error has slope ~1 and R^2 0.90-0.94; with one critical error ratio per system fitted on other
  designs, **10 of 12 per-tensor radii** are predicted within a factor of 2 (10-11 of 12 whichever designs fit
  the ratio). Limits: the ratio is system- and width-specific, and Rössler's is poorly determined.
* **Practical tests.** Protecting only the output layer from real pruning/quantization often improves valid
  prediction time but rarely prevents failure. Theory-guided mixed-precision bit allocation is a good *screen*
  (AUC 0.77-0.86 for predicting failure; random allocations are much worse) but its optimum is close to
  uniform bit widths and gives only small gains.
* **From the original matched-control study.** Against isotropic noise of the same norm, structured
  perturbations have longer valid prediction time (17/20 comparisons) and lower one-step error (18/20);
  matching per-tensor norms leaves 5/20 and 5/20 significant. Parent-relative one-step error predicts valid
  prediction time loss best (cross-validated R^2 0.77 -> 0.85 Lorenz, 0.74 -> 0.88 Rössler); long-run
  Lyapunov-exponent change is poorly predicted by everything tested.

Scope: three systems (two 3-D, one 6-D), small MLPs (about 4.6-5.0 thousand parameters; widths 32-128 on the
3-D systems), post-training compression only. See the paper (`paper/main.tex`, compiled `paper/main.pdf`) for
limitations; the new analyses are labelled exploratory there.

---

## Install and quick checks

```bash
pip install -r requirements.txt
python tests/test_core.py        # 8 correctness tests (gradients, Jacobians, matched norms, ...)
python sanity_check.py           # Jacobian and first-order-response checks (~1 min)
```
Python 3.10+ (developed and tested with 3.12). A LaTeX installation (`pdflatex`, `bibtex`) is needed only to
compile the paper.

---

## Reproduce the results

All raw per-model results used in the paper are included, so you can regenerate every table, figure and
number **without** re-running any experiment (Option A), or recompute everything from scratch (Option B).

### Option A: analysis only (minutes)

```bash
# 1. original 3-D matched-control study (figures/tables of Sections 4.1-4.4, 4.11)
python analyze.py          --results results_full_30seeds
python analyze_paired.py   --results results_full_30seeds
python analyze_extra.py    --results results_full_30seeds
python analyze_h1.py       --results results_full_30seeds --out paper
python make_paper_assets.py --results results_full_30seeds --out paper

# 2. extension: tensor interventions, Lorenz-96, protection experiment (writes fig_tensor, tab_tensor*, ...)
python analyze_extension.py --tensor results_tensor --l96 results_l96 --protect results_protect --out paper
# 3. first-order theory
python analyze_theory.py   --theory results_theory --out paper
# 4. mixed-precision quantization
python analyze_mixed.py    --mixed results_mixed --out paper
# 5. network width
python analyze_width.py    --theory results_theory --width results_width --out paper

# 6. assemble the manuscript (every number in the new sections is read from the files above)
python build_v2.py                                   # paper/main_orig.tex -> paper/main.tex
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Steps 2-5 take about ten minutes in total on one core. `paper/main_orig.tex` is the manuscript source of the
original two-system study; `build_v2.py` rewrites its title, abstract, summary table and relevant sections and
inserts the new sections. The generated `paper/main.tex` is also committed, so you can compile it directly.

This path was verified from a clean copy of the repository with all derived paper files deleted: steps 2-6
regenerated a byte-identical `paper/main.tex` and the same 31-page PDF.

Stand-alone summary of the tensor interventions (not needed for the paper):
`python analyze_tensor.py --results results_tensor`.

### Option B: recompute everything (hours)

Times are approximate, for one CPU core. Every runner is **resumable**: finished (system, seed) files are
skipped when you start it again, so it is safe to interrupt and restart (run long jobs with `setsid nohup`).
If you recompute the original study into a different folder than `results_full_30seeds`, pass that folder via
`--results` (`--main` for `run_protect.py`, `run_theory.py`, `run_mixed.py`) to every later script.

```bash
# 0. original 3-D study: Lorenz-63 and Rössler, 30 seeds, 353 perturbed models per parent      (~1 h)
python run_experiment.py --systems lorenz rossler --seeds 30 --K 10 --out results
python recompute_extra.py --results results            # rebuild parents; per-tensor displacements (~10 min)
# (then the analysis commands of Option A, step 1, with --results results)

# 1. tensor-isolated interventions on the 3-D parents (650 models per valid parent)              (~20 min)
python run_tensor_intervention.py --system lorenz  --K 10 --out results_tensor
python run_tensor_intervention.py --system rossler --K 10 --out results_tensor

# 2. Lorenz-96 (N=6, F=8): full matched-control design + tensor interventions, 30 seeds          (~35 min)
python run_l96.py --seeds 30 --K 10 --out results_l96

# 3. protection experiment: compress all / all but output layer / only output layer              (~25 min)
for s in lorenz rossler lorenz96; do python run_protect.py --system $s --out results_protect; done

# 4. first-order theory: analytic sensitivities + predictions for every intervention model       (~20 min)
python run_theory.py --out results_theory

# 5. mixed-precision quantization: uniform vs theory-guided vs 20 random allocations             (~25 min)
for s in lorenz rossler lorenz96; do python run_mixed.py --system $s --out results_mixed; done

# 6. network width: new parents with hidden width 32 and 128, 16 seeds                            (~40 min)
for H in 32 128; do for s in lorenz rossler; do
  python run_width.py --system $s --H $H --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --out results_width/H$H
done; done
```
Then run the analysis commands of Option A (steps 2-6). Dependencies between steps: 3, 4 and 5 need step 2
(`results_l96`, for the Lorenz-96 parents); 4 needs steps 1 and 2; 1, 3, 5 and 6 need the original study's
`config.json`, `reference_lle.json` and per-parent `raw/` files (for parent validity and the Lyapunov-time scale).

Fast smoke test of the machinery (a few minutes): `python run_experiment.py --systems lorenz --seeds 2 --K 3
--epochs 10 --out results_smoke` and `python run_l96.py --seeds 1 --K 2 --epochs 10 --out results_l96_smoke`.

---

## Which script produces which part of the paper

| Paper section | Content | Run | Analyse | Results folder |
|---|---|---|---|---|
| 4.1-4.4 | failure rates, robustness radius, H1 test, matched-norm comparisons | `run_experiment.py`, `recompute_extra.py` | `analyze.py`, `analyze_paired.py`, `analyze_extra.py`, `analyze_h1.py`, `make_paper_assets.py` | `results_full_30seeds/` |
| 4.5 | tensor-isolated interventions (Fig. 5, Tables 7-9) | `run_tensor_intervention.py`, `run_l96.py` | `analyze_extension.py` (`analyze_tensor.py`) | `results_tensor/`, `results_l96/tensor_seed*.csv` |
| 4.6 | first-order theory, critical error ratio, radii (Eqs. 1-2, Fig. 6, Table 12) | `run_theory.py` | `analyze_theory.py` | `results_theory/` |
| 4.7 | width generality (Table 13) | `run_width.py` | `analyze_width.py` | `results_width/H32`, `H128` (H=64 from `results_theory/`) |
| 4.8 | protecting the output layer (Table 14) | `run_protect.py` | `analyze_extension.py` | `results_protect/` |
| 4.9 | mixed-precision quantization (Table 15) | `run_mixed.py` | `analyze_mixed.py` | `results_mixed/` |
| 4.10 | Lorenz-96 replication (Table 16) | `run_l96.py` | `analyze_extension.py` | `results_l96/` |
| 4.11 | predictors of damage, VPT-MSE law | (original study) | `analyze_extra.py`, `make_paper_assets.py` | `results_full_30seeds/` |

Numbers quoted in the text of the new sections are read from `paper/ext_numbers.json`,
`theory_numbers.json`, `mixed_numbers.json` and `width_numbers.json` by `build_v2.py`; nothing in those
sections is typed by hand. (`analyze_mixed.py` also writes `figures/fig_mixed.pdf`, which the paper does not use.)

---

## Repository layout

```
jacsens/
  systems.py       Lorenz-63 / Rössler, RK4 integration, reference Lyapunov exponent
  model.py         residual MLP, manual backprop + Adam, analytic and batched Jacobians
  perturb.py       pruning, quantization, matched random controls
  dynamics.py      batched LLE, valid prediction time, sliced Wasserstein distance
  sensitivity.py   trajectory-weighted Jacobian-change measures, first-order LLE response
tests/test_core.py

# original study
run_experiment.py  recompute_extra.py  analyze.py  analyze_paired.py  analyze_extra.py  analyze_h1.py
make_paper_assets.py  sanity_check.py

# extension
run_tensor_intervention.py   noise confined to one tensor group (designs: total, leave-one-out, own-norm)
run_l96.py                   Lorenz-96: matched-control design + tensor interventions
run_protect.py               compress all / all-but-output / only-output tensors
run_theory.py                analytic sensitivities s_k and predicted one-step error per model
run_mixed.py                 mixed-precision quantization (uniform / theory-guided / random)
run_width.py                 hidden-width generality (H = 32, 128)
analyze_tensor.py  analyze_extension.py  analyze_theory.py  analyze_mixed.py  analyze_width.py
build_v2.py                  assembles paper/main.tex from paper/main_orig.tex and the result files

results_full_30seeds/  results_tensor/  results_l96/  results_protect/  results_theory/
results_mixed/         results_width/
paper/{main_orig.tex,main.tex,main.pdf,refs.bib,figures,tables,*.json}
```

---

## Result file formats

All results are plain CSV/JSON, one file per (system, seed), written when that seed finishes.

* `results_full_30seeds/raw/{system}_seed{k}.csv`: one row per perturbed model (parent first). Columns:
  `system seed role family level group control_of draw rel_norm mse lle lle_h1 lle_h2 alive collapsed sw1 vpt
  vpt_lt sens_orbit S_F S_F_rel S_2 S_lin S_lin0 parent_orbit_lle`. `role` is `parent`, `primary`, `control`
  or `grid`.
* `results_l96/main_seed{k}.csv`: Lorenz-96 matched-control design with columns
  `system seed role family level group control_of draw rel_norm mse mse_parent diverged collapsed vpt_lt
  parent_vpt_lt` (no per-model Lyapunov exponent); `info_seed{k}.json`: parent MSE, parent exponent, validity.
* `results_tensor/{system}_seed{k}.csv`, `results_l96/tensor_seed{k}.csv`: `design` (`total` / `loo` / `own`),
  `group` (`W1`, `W2`, `W3`, `biases`, `all`, `all_but_*`), `r` (noise level), `draw`, `rel_norm`, `r_W3`,
  `log10_mse_ratio`, `diverged`, `collapsed`, `fail`, `vpt_lt`, `vpt_loss`, `parent_vpt_lt`.
* `results_theory/{system}_seed{k}.csv`: the tensor-intervention rows plus `dmse_pred` (predicted MSE increase),
  `dmse_meas`, `noise_norm`, `mse_parent`, per-tensor sensitivities `S_*`, norms `norm_*` and sizes `n_*`.
* `results_protect/*.csv`: `family`, `level`, `scope` (`all` / `protOut` / `onlyOut`), `rel_norm`,
  `log10_mse_ratio`, failure and VPT columns.
* `results_mixed/*.csv`: `budget` (average bits), `scheme` (`uniform` / `theory` / `random`), `draw`, per-tensor
  bits `bits_*`, `avg_bits`, `pred_dmse`, failure and VPT columns.
* `results_width/H{32,128}/*.csv`: same format as `results_theory` (plus `H`) for the new parents.

Definitions used everywhere: a model **diverges** if `|x|_inf > 50` (standardised units) or becomes non-finite
and **collapses** if the standard deviation of its rollout falls below 0.05; **valid prediction time** (VPT) is the
first time the normalised error exceeds 0.4, in Lyapunov times (reference exponents 0.903, 0.0655 and 1.040 for
Lorenz-63, Rössler and Lorenz-96); **parent validity**: the parent's own rollout must not fail and, in the
original study and for Lorenz-96, its two half-run Lyapunov estimates must agree within 0.25 (the width
experiment applies only the first condition).

---

## Reproducibility notes

* **Deterministic.** Every parent, perturbation set and bootstrap uses a fixed seed path (seeds 0-29 per system;
  `run_width.py` uses seeds 0-15 and includes the width in its seed). Re-running a script regenerates the same
  files with the same NumPy version. Runners that rebuild parents from the original study
  (`recompute_extra.py`, `run_tensor_intervention.py`, `run_protect.py`, `run_theory.py`, `run_mixed.py`)
  assert that the rebuilt parent matches the stored parent MSE.
* **Unit of replication is the parent (seed).** Comparisons are paired per parent, summarised by medians with
  95% bootstrap intervals, Wilcoxon signed-rank tests and Holm adjustment within metric families.
* **Exploratory vs pre-specified.** The primary analyses of the original study (failure rates, matched-pair
  comparisons, predictor comparison) were specified before the full run. The robustness radius, the H1 test, the
  tensor interventions, the theory, the width check and the mixed-precision test were added later and are
  labelled exploratory in the paper.
* **Width experiment.** `results_width/` uses new parents trained with the same recipe at another width. Its
  validity rule is weaker, and the critical error ratio `c` is fitted on the total-norm design only, so the
  `H = 64` row of Table 13 differs from Table 12 (which also uses the leave-one-out design).

---

## Troubleshooting

* **`AssertionError` while rebuilding a parent**: the rebuilt parent does not match the stored MSE, usually
  because the NumPy version or the `config.json` differs from the one used for the stored results.
* **Long runs stopped by a timeout or a closed terminal**: start the same command again; finished seeds are
  skipped. `setsid nohup python ... &` keeps a job alive after the terminal closes.
* **LaTeX `font expansion` error**: the build loads `microtype` with `expansion=false`; keep that option if you
  edit the preamble. Tables and figures use `[tbp]` placement and the summary table occupies its own page.
* **Missing `results_l96/info_seed*.json`** when running `run_protect.py --system lorenz96` or `run_theory.py`:
  run `run_l96.py` first.

---

## License and citation

Code: MIT (see `LICENSE`). Please cite as in `CITATION.cff`. Archived release:
[doi:10.5281/zenodo.23157081](https://doi.org/10.5281/zenodo.23157081).
