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

## License

MIT (see `LICENSE`).
