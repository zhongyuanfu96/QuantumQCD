# QuantumQCD — code for "Quantum Quickest Change Detection"

Zhongyuan Fu, Jeremy Johnston, Xiaodong Wang, Luca Venturino. npj Quantum Information (in revision, 2026).

This repository contains the code that implements the measurement designs of the paper and reproduces its 15 figures:

- maximum-KL von Neumann measurement for a known post-change state (projected gradient ascent, Algorithm 1),
- maximum-sensitivity measurement for an unknown post-change state (closed form, Theorem 2) with the window-based CUSUM-like test,
- joint probe-state and measurement design for quantum channel change detection (rank minimization + alternating optimization, Algorithm 2),
- the robustness study with a depolarized pre-change state.

## Layout

| Path | Content |
|---|---|
| `src/quantum_qcd/quantum_states.py` | random pre-/post-change state ensembles (Wishart, unitary-product, rank-deficient) |
| `src/quantum_qcd/quantum_utils.py` | POVMs (maximum-sensitivity, standard, Helstrom), outcome distributions, depolarization |
| `src/quantum_qcd/quantum_detection.py` | CUSUM and window-based CUSUM-like tests; FAP/ADD simulation (CPU, optional GPU) |
| `src/quantum_qcd/pga.py` | projected gradient ascent for the maximum-KL measurement (full-rank and rank-deficient pre-change states) |
| `src/quantum_qcd/cusum_nb.py` | Monte-Carlo CUSUM routines used by the rank-deficient and channel experiments (stopping-cause split, augmented null outcome) |
| `src/quantum_qcd/channel.py` | quantum channel change detection: Kraus channels, log-det rank minimization, joint alternating optimization (known and unknown post-change channel) |
| `scripts/figstyle.py` | shared figure style (journal rules: no in-image titles, bold panel letters, 330 dpi PNG + vector PDF) |
| `scripts/fig01_*.py` ... `scripts/fig15_*.py` | one script per figure of the paper (see the table below) |
| `scripts/noise_study.py` | the depolarized pre-change state study behind Fig. 13 |
| `data/` | state sets and saved simulation outputs used by the figures (15 MB); `data/derived/` holds the caches written by the figure scripts |
| `tests/` | consistency tests (analytic FAP cap, vectorized vs. reference implementation) |
| `figures/` | output folder for `FigN.png` / `FigN.pdf` |

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy, matplotlib, torch (CPU)
export PYTHONPATH=src
python scripts/fig04_add_vs_fap_full_rank.py            # writes figures/Fig4.png and Fig4.pdf
python scripts/fig12_add_vs_fap_conditional_unknown.py  # about 8 minutes: Monte-Carlo simulation
python scripts/fig12_add_vs_fap_conditional_unknown.py --plot-only   # re-plot from data/derived/ without simulating
```

Every script accepts `--out-dir` (default `figures/`) and `--plot-only` (re-use the cache in `data/derived/` if present). Set `OMP_NUM_THREADS=2` when running several scripts at once.

## Figures

Runtimes are for one run on a laptop CPU (Apple silicon, two threads). "Saved data" means the script only plots arrays produced for the paper; "simulation" means the script recomputes the plotted quantities (seeded, so re-runs are reproducible; Monte-Carlo figures differ from the published ones only by sampling noise).

| Figure | Script | Input | Runtime |
|---|---|---|---|
| 1 | `fig01_pga_convergence_full_rank.py` | saved data: PGA objective histories, `data/plot_alg_convergence/`, `data/plot_3pair_state/` | 1 s |
| 2 | `fig02_eigenvalue_histograms.py` | simulation: eigenvalues of 10^4 Wishart and unitary-product states (seed 0) | 1 s |
| 3 | `fig03_kl_barplot.py` | saved data: KL divergences of 500 state pairs per ensemble, `data/plot_N8_100_pairs/` (Wishart), `data/adap_grad_jeremy/plot_N8_500_pairs/` (unitary-product) | 1 s |
| 4 | `fig04_add_vs_fap_full_rank.py` | saved data: ADD/FAP sweeps of the same 500 pairs | 1 s |
| 5 | `fig05_pga_convergence_rank_deficient.py` | simulation: Algorithm 1 for pre-change ranks 5-8 (deterministic) | 2 s |
| 6 | `fig06_add_vs_fap_rank_deficient.py` | simulation: Algorithm 1 + CUSUM Monte Carlo, maximum-KL vs. Helstrom | 40 s |
| 7 | `fig07_add_vs_fap_conditional_known.py` | simulation: CUSUM Monte Carlo split by stopping cause (50 000 runs per threshold) | 5 min |
| 8 | `fig08_sensitivity_barplot.py` | closed-form sensitivities of the 500 pre-change states per ensemble | 1 s |
| 9 | `fig09_windows.py` | saved data: window-based CUSUM-like test sweeps, `data/window_u/` | 1 s |
| 10 | `fig10_estimated_vs_exact.py` | saved data: estimated vs. exact post-change distribution, `data/window_u/` | 1 s |
| 11 | `fig11_add_vs_fap_rank_deficient_unknown.py` | simulation: maximum-sensitivity measurement, CUSUM Monte Carlo for ranks 5-8 | 18 s |
| 12 | `fig12_add_vs_fap_conditional_unknown.py` | simulation: CUSUM Monte Carlo split by stopping cause (5 million runs per estimator and rank) | 8 min |
| 13 | `fig13_noisy_pre_change.py` | `noise_study.py` with the caches in `data/noise/` (depolarized pre-change state) | 45 s |
| 14 | `fig14_channel_convergence.py` | simulation: Algorithm 2 (rank minimization, joint optimization), deterministic seeds | 16 s |
| 15 | `fig15_channel_add_vs_fap.py` | simulation: channel designs of Fig. 14 + CUSUM Monte Carlo | 54 s |

Provenance: each script's docstring names the research notebook cell it was ported from; the published figures were traced to those cells by matching the notebooks' embedded output images byte for byte.

## License

MIT (see `LICENSE`).
