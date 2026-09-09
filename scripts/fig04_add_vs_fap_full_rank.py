#!/usr/bin/env python3
"""Figure 4: average detection delay (ADD) versus false alarm period (FAP) for
full-rank state pairs, under the maximum-KL, Helstrom and standard measurements.

Panel a: 500 Wishart state pairs (N = 8).
Panel b: 500 unitary-product state pairs (N = 8).

Source: notebooks_raw/Quantum_Anomaly_Detection/optimization_known/adaptive_gradient/
        plots_combined.ipynb, cell 13 (imports from cell 1).

The figure is plot-only: the twelve (500, 100) arrays of parametric (FAP, ADD)
curves were produced by the long simulation cells of adaptive_gradient.ipynb /
adaptive_gradient_jeremy.ipynb and are shipped in
    data/plot_N8_100_pairs/   (Wishart ensemble; the folder name says 100 but it
                               holds 500 pairs)
    data/plot_N8_500_pairs/   (unitary-product ensemble)
The only computation is, per state pair, the truncation of the curve at the first
threshold whose FAP reaches max_fap = 5000, a linear interpolation onto a common
FAP grid, and the mean / population standard deviation across the 500 pairs.

Usage:
    python3 scripts/fig04_add_vs_fap_full_rank.py [--out-dir DIR] [--plot-only]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# quantum_qcd is not needed for this figure (no simulation), but the release
# scripts all expose the package on sys.path so they can be run from anywhere.
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from figstyle import new_figure, label_panels, save_figure  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA_DIR, "derived", "fig04_add_vs_fap.npz")

# --- parameters of the source cell (unchanged) -------------------------------
MAX_FAP = 5000
N_POINTS = 200          # resolution of the common FAP grid
N = 8                   # Hilbert-space dimension of the stored runs
N_SETS = 500            # state pairs per ensemble
MEASUREMENTS = ["vonNeumann", "helstrom", "standard"]

# panel key -> (sub-folder under data/, manuscript description)
PANELS = [
    ("wishart", "plot_N8_100_pairs"),
    ("unitary", "plot_N8_500_pairs"),
]
# plot order, legend labels and colours exactly as in the notebook cell:
# 'Maximum-KL' and 'Helstrom' take the first two entries of the default cycle
# (C0 tab:blue, C1 tab:orange); 'Standard' asks for color='g' (pure green).
LINES = [
    ("vonNeumann", "Maximum-KL", None),
    ("helstrom", "Helstrom", None),
    ("standard", "Standard", "g"),
]


def curves_for_panel(folder):
    """Mean and std of the interpolated ADD-vs-FAP curves for one ensemble."""
    d = os.path.join(DATA_DIR, folder)
    fap = {m: np.load(os.path.join(d, f"N{N}_{m}_FAP.npy")) for m in MEASUREMENTS}
    add = {m: np.load(os.path.join(d, f"N{N}_{m}_ADD.npy")) for m in MEASUREMENTS}

    common_x = np.linspace(0, MAX_FAP, N_POINTS)
    out = {}
    for m in MEASUREMENTS:
        interp = np.zeros((N_SETS, N_POINTS))
        for i in range(N_SETS):
            # first threshold index whose FAP reaches max_fap; the curve is cut
            # there before interpolating (load-bearing: without the cut, np.interp
            # sees the far tail of the parametric curve and the mean changes).
            # Every row of the shipped arrays does reach max_fap (last column
            # 6597-19418), so no row is flat-extrapolated here.
            idx = np.where(fap[m][i, :] >= MAX_FAP)[0][0]
            interp[i, :] = np.interp(common_x, fap[m][i, :idx + 1], add[m][i, :idx + 1])
        out[m + "_mean"] = np.mean(interp, axis=0)
        out[m + "_std"] = np.std(interp, axis=0)  # np.std default ddof=0
    return common_x, out


def compute():
    common_x = None
    data = {}
    for key, folder in PANELS:
        common_x, out = curves_for_panel(folder)
        for name, arr in out.items():
            data[f"{key}_{name}"] = arr
        print(f"{key:8s} mean ADD at FAP={MAX_FAP}: "
              f"vN={out['vonNeumann_mean'][-1]:.4f}, "
              f"helstrom={out['helstrom_mean'][-1]:.4f}, "
              f"standard={out['standard_mean'][-1]:.4f}")
    data["common_x"] = common_x
    return data


def load_cache():
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


def make_figure(data, out_dir):
    fig, axes = new_figure(nrows=1, ncols=2, width="double", sharex=True, sharey=True)
    common_x = data["common_x"]

    for ax, (key, _folder) in zip(axes, PANELS):
        for meas, label, color in LINES:
            mean = data[f"{key}_{meas}_mean"]
            std = data[f"{key}_{meas}_std"]
            kw = {} if color is None else {"color": color}
            line = ax.plot(common_x, mean, label=label, **kw)[0]
            ax.fill_between(common_x, mean - std, mean + std,
                            alpha=0.2, color=line.get_color())
        ax.set_xlim(0, MAX_FAP)
        ax.set_yscale("log")
        ax.grid(linestyle="--")

    # shared axis labels as in the source cell; the bottom strip of the figure is
    # reserved for the shared x label and the three-column plot key below the axes
    # (the notebook put the key there too, and it stays legible at 6.5 in).
    fig.supylabel("ADD", fontsize=8)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.supxlabel("FAP", fontsize=8, y=0.110)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=3, frameon=False)
    label_panels(fig, axes)
    return save_figure(fig, 4, out_dir=out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--plot-only", action="store_true",
                    help="re-plot from data/derived/fig04_add_vs_fap.npz")
    args = ap.parse_args()

    if args.plot_only and os.path.exists(CACHE):
        data = load_cache()
        print(f"loaded cached curves from {CACHE}")
    else:
        data = compute()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, **data)
        print(f"cached curves to {CACHE}")

    png = make_figure(data, args.out_dir)
    print(f"wrote {png} and {png[:-4]}.pdf")


if __name__ == "__main__":
    main()
