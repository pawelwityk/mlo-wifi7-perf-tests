#!/usr/bin/env python3
"""
plot_results.py -- Latency Scenario #4B (4.2.5)

Reads per-run CSVs from thr_runs_4_2_5/ (produced by run_all_rerun.py) and
generates three plots:

  plots/4_2_5.svg             -- mean latency & throughput vs offered load
  plots/4_2_5_pct_vs_load.svg -- p50/p99 percentile band vs offered load
  plots/4_2_5_bars.svg        -- grouped-bar (p50/p95/p99) at selected loads

File naming convention (runner output):
  thr_runs_4_2_5/M{mode}_Load{load}_rng{rng}_result.csv
  content: lat,p50,p95,p99,thr
"""

# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim

import os
import re
import csv
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR    = "thr_runs_4_2_5"
PLOT_DIR      = "plots"
PLOT_FILE     = os.path.join(PLOT_DIR, "4_2_5.svg")
PLOT_PCT      = os.path.join(PLOT_DIR, "4_2_5_pct_vs_load.svg")
PLOT_BARS     = os.path.join(PLOT_DIR, "4_2_5_bars.svg")

STR_MODES     = ["SLO", "STR2", "EMLSR2", "STR1P1", "STR5"]
OFFERED_LOADS = list(range(100, 2600, 100))
BAR_LOADS     = [100, 500, 1000, 2000, 2500]

T95_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

MODE_COLORS = {
    "SLO":    "#2E75D8",
    "STR2":   "#E63B0E",
    "EMLSR2": "#9C27B0",
    "STR1P1": "#28A745",
    "STR5":   "#F0B400",
}
MODE_LABELS = {
    "SLO":    "SLO (1 link)",
    "STR2":   "MLO-STR:2 (2 links)",
    "EMLSR2": "MLO-EMLSR:2 (2 links)",
    "STR1P1": "MLO-STR:1+1 (2 links)",
    "STR5":   "MLO-STR:5 (5 links)",
}
MODE_MARKERS = {
    "SLO":    "o",
    "STR2":   "s",
    "EMLSR2": "D",
    "STR1P1": "^",
    "STR5":   "P",
}

os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def _parse_row(text):
    """Return (lat, p50, p95, p99, thr) floats or None."""
    reader = csv.reader([text])
    row = next(reader, [])
    parts = [p.strip() for p in row if p.strip()]
    if len(parts) >= 5:
        return tuple(float(p) for p in parts[:5])
    if len(parts) >= 4:
        lat, p50, p99, thr = (float(p) for p in parts[:4])
        return lat, p50, p99, p99, thr
    if len(parts) >= 2:
        lat, thr = float(parts[0]), float(parts[1])
        return lat, lat, lat, lat, thr
    return None


def load_results():
    """Return dict keyed by (mode, load) -> [(rng, lat, p50, p95, p99, thr)]."""
    data = {}
    if not os.path.isdir(OUTPUT_DIR):
        return data
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        m = re.match(r"M(\w+)_Load(\d+)_rng(\d+)_result\.csv$", fname)
        if not m:
            continue
        mode, load, rng = m.group(1), int(m.group(2)), int(m.group(3))
        if mode not in STR_MODES:
            continue
        fpath = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(fpath) as f:
                parsed = _parse_row(f.read().strip())
            if parsed is None:
                continue
            lat, p50, p95, p99, thr = parsed
            key = (mode, load)
            data.setdefault(key, [])
            data[key].append((rng, lat, p50, p95, p99, thr))
        except Exception:
            pass
    return data


# ---------------------------------------------------------------------------
#  Statistics helpers
# ---------------------------------------------------------------------------

def ci95(values):
    if len(values) <= 1:
        return 0.0
    n = len(values)
    t = T95_CRITICAL.get(n - 1, 1.96)
    return t * stdev(values) / (n ** 0.5)


def remove_outliers(values):
    """Two-stage IQR + MAD outlier removal (delegates to _common/_robust)."""
    from _robust import remove_outliers_robust
    return remove_outliers_robust(values)


# ---------------------------------------------------------------------------
#  Plot 1: mean latency (+ p99 dashed) and throughput vs offered load
# ---------------------------------------------------------------------------

def plot_line(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Latency Scenario #4B — DL Latency & Throughput vs. Offered Load")

    for mode in STR_MODES:
        xs, lat_m, lat_e, p99_m, p99_e, thr_m, thr_e = [], [], [], [], [], [], []
        for load in OFFERED_LOADS:
            rows = data.get((mode, load), [])
            if not rows:
                continue
            lats = remove_outliers([r[1] for r in rows])
            p99s = remove_outliers([r[4] for r in rows])
            thrs = remove_outliers([r[5] for r in rows])
            xs.append(load)
            lat_m.append(mean(lats));   lat_e.append(ci95(lats))
            p99_m.append(mean(p99s));   p99_e.append(ci95(p99s))
            thr_m.append(mean(thrs));   thr_e.append(ci95(thrs))

        if not xs:
            continue

        c  = MODE_COLORS[mode]
        mk = MODE_MARKERS[mode]
        axes[0].errorbar(xs, lat_m, yerr=lat_e, label=f"{MODE_LABELS[mode]} mean",
                         color=c, marker=mk, linestyle="-", markersize=4,
                         capsize=3, linewidth=1.6)
        axes[0].errorbar(xs, p99_m, yerr=p99_e, label=f"{MODE_LABELS[mode]} p99",
                         color=c, marker=mk, linestyle="dotted",
                         markersize=4, capsize=3, linewidth=1.2)
        axes[1].errorbar(xs, thr_m, yerr=thr_e, label=MODE_LABELS[mode],
                         color=c, marker=mk, linestyle="-", markersize=4,
                         capsize=3, linewidth=1.6)

    for ax, ylabel in zip(axes, ["DL Latency (ms)", "Aggregate Throughput [Mbit/s]"]):
        ax.set_xlabel("Total Offered Load [Mbit/s]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    print(f"[plot] saved -> {PLOT_FILE}", flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Plot 2: p50 / p99 percentile band vs offered load (one sub-panel per mode)
# ---------------------------------------------------------------------------

def plot_percentile_bands(data):
    ncols = len(STR_MODES)
    if ncols <= 3:
        nrows, ncols2 = 1, ncols
    else:
        ncols2 = 3
        nrows = int(np.ceil(ncols / ncols2))
    fig, axes = plt.subplots(nrows, ncols2,
                             figsize=(4.5 * ncols2, 3.6 * nrows),
                             sharey=False)
    if ncols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).reshape(-1)
    fig.suptitle("Latency Scenario #4B — p50 / p99 vs. Offered Load")

    for ax, mode in zip(axes, STR_MODES):
        xs, p50_m, p50_e, p99_m, p99_e = [], [], [], [], []
        for load in OFFERED_LOADS:
            rows = data.get((mode, load), [])
            if not rows:
                continue
            xs.append(load)
            p50s = remove_outliers([r[2] for r in rows])
            p99s = remove_outliers([r[4] for r in rows])
            p50_m.append(mean(p50s));  p50_e.append(ci95(p50s))
            p99_m.append(mean(p99s));  p99_e.append(ci95(p99s))

        if xs:
            c = MODE_COLORS[mode]
            xs_arr = np.array(xs)
            p50_arr, p50_err = np.array(p50_m), np.array(p50_e)
            p99_arr, p99_err = np.array(p99_m), np.array(p99_e)
            ax.plot(xs_arr, p50_arr, color=c, linestyle="-",  marker="o", markersize=3,
                    linewidth=1.4, label="p50")
            ax.fill_between(xs_arr, p50_arr - p50_err, p50_arr + p50_err,
                            color=c, alpha=0.15)
            ax.plot(xs_arr, p99_arr, color=c, linestyle="--", marker="s", markersize=3,
                    linewidth=1.4, label="p99")
            ax.fill_between(xs_arr, p99_arr - p99_err, p99_arr + p99_err,
                            color=c, alpha=0.15)
            ax.fill_between(xs_arr, p50_arr, p99_arr, color=c, alpha=0.08)

        ax.set_title(MODE_LABELS[mode], fontsize=9)
        ax.set_xlabel("Offered Load [Mbit/s]")
        ax.set_ylabel("Channel access delay [ms]")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(PLOT_PCT, format="svg")
    print(f"[plot] saved -> {PLOT_PCT}", flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Plot 3: grouped bars (p50 / p95 / p99) at selected loads
# ---------------------------------------------------------------------------

def plot_bars(data):
    bar_loads = [lo for lo in BAR_LOADS if lo in OFFERED_LOADS]
    n_loads   = len(bar_loads)
    n_modes   = len(STR_MODES)

    fig, axes = plt.subplots(1, n_loads, figsize=(3.5 * n_loads, 5), sharey=False)
    if n_loads == 1:
        axes = [axes]
    fig.suptitle("Latency Scenario #4B — Percentile Latency at Selected Loads")

    bar_width = 0.15
    x = np.arange(n_modes)

    for ax, load in zip(axes, bar_loads):
        for pi, (pct_name, col, alpha) in enumerate(zip(["p50", "p95", "p99"], [2, 3, 4], [1.0, 0.65, 0.35])):
            vals = []
            errs = []
            for mode in STR_MODES:
                rows = data.get((mode, load), [])
                if rows:
                    vs = remove_outliers([r[col] for r in rows])
                    vals.append(mean(vs))
                    errs.append(ci95(vs))
                else:
                    vals.append(0.0)
                    errs.append(0.0)

            ax.bar(x + pi * bar_width, vals, bar_width,
                   yerr=errs, capsize=3,
                   color=[MODE_COLORS[m] for m in STR_MODES],
                   alpha=alpha,
                   label=pct_name)

        ax.set_title(f"Load = {load} Mb/s")
        ax.set_xlabel("Mode")
        ax.set_ylabel("Channel access delay [ms]")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([MODE_LABELS[m].split(" ")[0] for m in STR_MODES],
                           fontsize=8, rotation=20, ha="right")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_BARS, format="svg")
    print(f"[plot] saved -> {PLOT_BARS}", flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    data = load_results()
    total = sum(len(v) for v in data.values())
    print(f"[*] Loaded {total} run results from {OUTPUT_DIR}/", flush=True)

    if total == 0:
        print("[!] No results found — run run_all_rerun.py first.", flush=True)
        return

    plot_line(data)
    plot_percentile_bands(data)
    plot_bars(data)

    print("[*] All plots generated.", flush=True)


if __name__ == "__main__":
    main()
