#!/usr/bin/env python3
"""
plot_results.py -- Latency Scenario #4A (4.2.4)

Reads per-run CSVs from thr_runs_4_2_4/ (produced by run_all_rerun.py) and
generates three plots:

  plots/4_2_4.svg             -- mean latency & throughput vs offered load
  plots/4_2_4_pct_vs_load.svg -- p50/p99 percentile band vs offered load
  plots/4_2_4_bars.svg        -- grouped-bar (p50/p95/p99) at selected loads

File naming convention (runner output):
  thr_runs_4_2_4/L{nl}_Load{load}_rng{rng}_result.csv
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
OUTPUT_DIR    = "thr_runs_4_2_4"
PLOT_DIR      = "plots"
PLOT_FILE     = os.path.join(PLOT_DIR, "4_2_4.svg")
PLOT_PCT      = os.path.join(PLOT_DIR, "4_2_4_pct_vs_load.svg")
PLOT_BARS     = os.path.join(PLOT_DIR, "4_2_4_bars.svg")

NUM_LINKS     = [1, 2, 4]
OFFERED_LOADS = list(range(100, 2600, 100))
RUNS          = 20
BAR_LOADS     = [100, 1000, 2500]   # 0.1 / 1 / 2.5 Gbps (matches paper)

T95_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

MODE_COLORS  = {1: "#2E75D8", 2: "#E63B0E", 4: "#F0B400"}
MODE_LABELS  = {1: "SL", 2: "STR:2", 4: "STR:4"}
MODE_MARKERS = {1: "o", 2: "s", 4: "^"}

os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def _parse_row(text):
    """Return (lat, p50, p95, p99, thr) or None."""
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


def load_all_results():
    """Return raw[(nl, load)] = [(rng, lat, p50, p95, p99, thr)]."""
    raw = {}
    if not os.path.isdir(OUTPUT_DIR):
        print(f"[warn] Output directory not found: {OUTPUT_DIR}")
        return raw
    pattern = re.compile(r"^L(\d+)_Load(\d+)_rng(\d+)_result\.csv$")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        m = pattern.match(fname)
        if not m:
            continue
        nl, load, rng = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if nl not in NUM_LINKS:
            continue
        try:
            with open(os.path.join(OUTPUT_DIR, fname)) as f:
                parsed = _parse_row(f.read().strip())
            if parsed is None:
                continue
            lat, p50, p95, p99, thr = parsed
            raw.setdefault((nl, load), [])
            raw[(nl, load)].append((rng, lat, p50, p95, p99, thr))
        except Exception as exc:
            print(f"[warn] Could not read {fname}: {exc}")
    return raw


# ---------------------------------------------------------------------------
#  Statistics helpers
# ---------------------------------------------------------------------------

def remove_outliers(values):
    """Two-stage IQR + MAD outlier removal (delegates to _common/_robust)."""
    from _robust import remove_outliers_robust
    return remove_outliers_robust(values)


def ci95(values):
    n = len(values)
    if n <= 1:
        return 0.0
    t = T95_CRITICAL.get(n - 1, 1.96)
    return t * stdev(values) / (n ** 0.5)


# ---------------------------------------------------------------------------
#  Plot A: mean latency (+ p99) and throughput vs offered load
# ---------------------------------------------------------------------------

def plot_line(raw):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Scenario 4.2.4 -- DL Latency & Throughput vs. Offered Load")

    for nl in NUM_LINKS:
        xs, lat_m, lat_e, p99_m, p99_e, thr_m, thr_e = [], [], [], [], [], [], []
        for load in OFFERED_LOADS:
            rows = raw.get((nl, load), [])
            if not rows:
                continue
            lats = remove_outliers([r[1] for r in rows])
            p99s = remove_outliers([r[4] for r in rows])
            thrs = remove_outliers([r[5] for r in rows])
            xs.append(load)
            lat_m.append(mean(lats)); lat_e.append(ci95(lats))
            p99_m.append(mean(p99s)); p99_e.append(ci95(p99s))
            thr_m.append(mean(thrs)); thr_e.append(ci95(thrs))

        if not xs:
            continue

        c, mk = MODE_COLORS[nl], MODE_MARKERS[nl]
        xs_arr  = np.array(xs)
        lat_arr = np.array(lat_m)
        lat_err = np.array(lat_e)
        p99_arr = np.array(p99_m)
        p99_err = np.array(p99_e)
        thr_arr = np.array(thr_m)
        thr_err = np.array(thr_e)

        # Mean latency line + 95% CI shaded band
        axes[0].plot(xs_arr, lat_arr, color=c, marker=mk, linestyle="-",
                     markersize=4, linewidth=1.6, label=f"{MODE_LABELS[nl]} mean")
        axes[0].fill_between(xs_arr, lat_arr - lat_err, lat_arr + lat_err,
                             color=c, alpha=0.18)
        # p99 line + 95% CI shaded band
        axes[0].plot(xs_arr, p99_arr, color=c, marker=mk, linestyle="dotted",
                     markersize=4, linewidth=1.2, label=f"{MODE_LABELS[nl]} p99")
        axes[0].fill_between(xs_arr, p99_arr - p99_err, p99_arr + p99_err,
                             color=c, alpha=0.10)
        # Throughput line + 95% CI shaded band
        axes[1].plot(xs_arr, thr_arr, color=c, marker=mk, linestyle="-",
                     markersize=4, linewidth=1.6, label=MODE_LABELS[nl])
        axes[1].fill_between(xs_arr, thr_arr - thr_err, thr_arr + thr_err,
                             color=c, alpha=0.18)

    for ax, ylabel in zip(axes, ["Channel access delay [ms]", "Aggregate Throughput [Mbit/s]"]):
        ax.set_xlabel("Total Offered Load [Mbit/s]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    print(f"[plot] saved -> {PLOT_FILE}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Plot B: p50-p99 percentile band vs offered load  (paper Fig. 4.10 style)
# ---------------------------------------------------------------------------

def plot_percentile_bands(raw):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_title("Scenario 4.2.4 -- Latency percentiles vs. Total Traffic Load")

    for nl in NUM_LINKS:
        xs, y50, y50_e, y99, y99_e = [], [], [], [], []
        for load in OFFERED_LOADS:
            rows = raw.get((nl, load), [])
            if not rows:
                continue
            xs.append(load)
            p50s = remove_outliers([r[2] for r in rows])
            p99s = remove_outliers([r[4] for r in rows])
            y50.append(mean(p50s));  y50_e.append(ci95(p50s))
            y99.append(mean(p99s));  y99_e.append(ci95(p99s))

        if not xs:
            continue

        c = MODE_COLORS[nl]
        lbl = MODE_LABELS[nl]
        xs_arr = np.array(xs)
        y50_arr, y50_err = np.array(y50), np.array(y50_e)
        y99_arr, y99_err = np.array(y99), np.array(y99_e)

        ax.plot(xs_arr, y50_arr, color=c, linestyle="--", linewidth=1.8,
                marker=MODE_MARKERS[nl], markersize=5,
                label=f"50%-tile {lbl}")
        ax.fill_between(xs_arr, y50_arr - y50_err, y50_arr + y50_err,
                        color=c, alpha=0.15)
        ax.plot(xs_arr, y99_arr, color=c, linestyle="-",  linewidth=2.2,
                marker=MODE_MARKERS[nl], markersize=5,
                label=f"99%-tile {lbl}")
        ax.fill_between(xs_arr, y99_arr - y99_err, y99_arr + y99_err,
                        color=c, alpha=0.15)
        ax.fill_between(xs_arr, y50_arr, y99_arr, color=c, alpha=0.08)

    # Reference tick marks matching paper x-axis: 100, 1000, 2500
    ax.set_xticks([100, 500, 1000, 1500, 2000, 2500])
    ax.set_xlabel("Total traffic load [Mbit/s]")
    ax.set_ylabel("Channel access delay [ms]")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(PLOT_PCT, format="svg")
    print(f"[plot] saved -> {PLOT_PCT}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Plot C: paper-style stacked bars (p50 / p95-p50 / p99-p95) at 0.1/1/2.5 Gbps
# ---------------------------------------------------------------------------

def plot_bars(raw):
    bar_loads  = [l for l in BAR_LOADS if any(raw.get((nl, l)) for nl in NUM_LINKS)]
    load_gbps  = [f"{l/1000:.1f}".rstrip('0').rstrip('.') for l in bar_loads]
    if not bar_loads:
        return

    n_loads = len(bar_loads)
    fig, ax = plt.subplots(figsize=(7, 5))
    x      = np.arange(n_loads)
    width  = 0.22
    offsets = {1: -width, 2: 0.0, 4: width}

    for nl in NUM_LINKS:
        p50_vals, p95_delta, p99_delta, p99_total, p99_ci = [], [], [], [], []
        for load in bar_loads:
            rows = raw.get((nl, load), [])
            if rows:
                p50 = mean(remove_outliers([r[2] for r in rows]))
                p95 = mean(remove_outliers([r[3] for r in rows]))
                p99s = remove_outliers([r[4] for r in rows])
                p99 = mean(p99s)
                err = ci95(p99s)
            else:
                p50 = p95 = p99 = 1e-3
                err = 0.0
            p50_vals.append(max(p50, 1e-3))
            p95_delta.append(max(0.0, p95 - p50))
            p99_delta.append(max(0.0, p99 - p95))
            p99_total.append(max(p99, 1e-3))
            p99_ci.append(err)

        xpos = x + offsets[nl]
        c    = MODE_COLORS[nl]
        # p50 segment (full opacity)
        ax.bar(xpos, p50_vals, width=width, color=c, edgecolor="black",
               linewidth=0.4, label=MODE_LABELS[nl], zorder=3)
        # p95 segment (medium opacity)
        ax.bar(xpos, p95_delta, width=width, bottom=p50_vals,
               color=c, alpha=0.65, edgecolor="black", linewidth=0.2, zorder=3)
        # p99 segment (low opacity)
        ax.bar(xpos, p99_delta, width=width,
               bottom=np.array(p50_vals) + np.array(p95_delta),
               color=c, alpha=0.35, edgecolor="black", linewidth=0.2, zorder=3)

        # Annotate p99 value above each bar, with 95% CI error whisker
        for xi, p99, err in zip(xpos, p99_total, p99_ci):
            top = p99 + err
            ax.errorbar(xi, p99, yerr=err, fmt="none", color="black",
                        capsize=4, linewidth=1.1, zorder=5)
            ax.text(xi, top * 1.25, f"{p99:.1f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:g} ms"
    ))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels(load_gbps)
    ax.set_xlabel("Total traffic load [Gbit/s]")
    ax.set_ylabel("Channel access delay: 50/95/99 percentile [ms]")
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.35, zorder=0)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout()
    plt.savefig(PLOT_BARS, bbox_inches="tight", format="svg")
    print(f"[plot] saved -> {PLOT_BARS}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    raw = load_all_results()
    total = sum(len(v) for v in raw.values())
    print(f"[info] Loaded {total} run results from {OUTPUT_DIR}/")

    if total == 0:
        print("[warn] No data found -- run run_all_rerun.py first.")
        return

    plot_line(raw)
    plot_percentile_bands(raw)
    plot_bars(raw)


if __name__ == "__main__":
    main()
