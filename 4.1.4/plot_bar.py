#!/usr/bin/env python3
"""
plot_bar.py — Throughput Scenario #3 (4.1.4)

Reads pre-collected CSVs from thr_runs_4_1_4/ and produces a grouped bar
chart.

X axis  : number of OBSS stations per link (0 – 10)
Y axis  : Main BSS MacRx throughput [Mbit/s]
Groups  : one group per OBSS count (x-tick)
Bars    : 6 bars per group  →  SLO DL, SLO UL, STR DL, STR UL, EMLSR DL, EMLSR UL
Error   : 95 % confidence interval half-width (t-distribution)
"""

# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR = "thr_runs_4_1_4"
PLOT_DIR   = "plots"
PLOT_FILE  = os.path.join(PLOT_DIR, "4_1_4.svg")

MODES       = ["SLO", "STR", "EMLSR"]
OBSS_COUNTS = list(range(11))   # 0 … 10
CI          = 0.95

# ── Palette: lighter shade = UL, darker shade = DL ────────────────────────
COLORS = {
    ("SLO",   "DL"): "#2b5c9e",
    ("SLO",   "UL"): "#7aaed6",
    ("STR",   "DL"): "#276f41",
    ("STR",   "UL"): "#72c294",
    ("EMLSR", "DL"): "#b84a0e",
    ("EMLSR", "UL"): "#f0a070",
}

HATCH = {"DL": "",  "UL": "//"}

# ── Helpers ────────────────────────────────────────────────────────────────

def load_results():
    """Return raw[mode]["DL"|"UL"][n_obss] = [values…].

    File format: {mode}_{direction}_n{N}_rng{R}_result.csv with a single
    throughput value [Mbit/s] per file (one direction per simulation run).
    """
    raw = {
        m: {"DL": {n: [] for n in OBSS_COUNTS},
            "UL": {n: [] for n in OBSS_COUNTS}}
        for m in MODES
    }
    pattern = re.compile(
        r"^([A-Za-z]+)_(DL|UL)_n(\d+)_rng(\d+)_result\.csv$"
    )
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        m = pattern.match(fname)
        if not m:
            continue
        mode, direction, n_obss = m.group(1), m.group(2), int(m.group(3))
        if mode not in MODES or n_obss not in OBSS_COUNTS:
            continue
        try:
            with open(os.path.join(OUTPUT_DIR, fname)) as f:
                thr = float(f.read().strip())
            raw[mode][direction][n_obss].append(thr)
        except Exception as exc:
            print(f"[warn] {fname}: {exc}")
    return raw


def ci_hw(data):
    """Return (mean, half-width of 95 % CI) after IQR+MAD outlier removal."""
    from _robust import ci_hw_robust
    return ci_hw_robust(data, confidence=CI)


# ── Plot ───────────────────────────────────────────────────────────────────

def plot(raw):
    DIRS  = ["DL", "UL"]
    n_scenarios = len(MODES) * len(DIRS)   # 6
    bar_w = 0.12                           # width of one bar
    group_w = n_scenarios * bar_w          # total width of one group
    # offsets of each bar relative to the group centre
    offsets = np.linspace(-group_w / 2 + bar_w / 2,
                           group_w / 2 - bar_w / 2,
                           n_scenarios)

    x = np.arange(len(OBSS_COUNTS))

    fig, ax = plt.subplots(figsize=(16, 6))

    bar_idx = 0
    for mode in MODES:
        for direction in DIRS:
            means, hws = [], []
            for n in OBSS_COUNTS:
                mn, hw = ci_hw(raw[mode][direction][n])
                means.append(mn)
                hws.append(hw)

            color = COLORS[(mode, direction)]
            hatch = HATCH[direction]
            ax.bar(
                x + offsets[bar_idx],
                means,
                width=bar_w,
                color=color,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.5,
                label=f"{mode} {direction}",
                zorder=3,
            )
            ax.errorbar(
                x + offsets[bar_idx],
                means,
                yerr=hws,
                fmt="none",
                color="black",
                capsize=3,
                capthick=1,
                elinewidth=1,
                zorder=4,
            )
            bar_idx += 1

    ax.set_xlabel("OBSS network load", fontsize=12)
    ax.set_ylabel("Throughput [Mbit/s]", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n/10:.1f}" for n in OBSS_COUNTS])
    ax.set_xlim(-0.5, len(OBSS_COUNTS) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Throughput Scenario #3 — Main BSS throughput vs. OBSS stations per link\n"
        "5 GHz (ch42) + 6 GHz (ch7), 80 MHz, EHT MCS 13  |  "
        f"Mean ± {int(CI*100)}% CI",
        fontsize=11,
    )

    # Custom legend: solid patch = DL, hatched = UL; colours encode mode
    legend_handles = []
    for mode in MODES:
        for direction, label_suffix in [("DL", "DL"), ("UL", "UL")]:
            p = mpatches.Patch(
                facecolor=COLORS[(mode, direction)],
                hatch=HATCH[direction],
                edgecolor="grey",
                linewidth=0.5,
                label=f"{mode} {label_suffix}",
            )
            legend_handles.append(p)

    ax.legend(handles=legend_handles, ncol=3, fontsize=9,
              title="Mode / Direction", title_fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    print(f"\n[OK] Plot saved to {PLOT_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_DIR):
        raise SystemExit(f"[error] Results directory not found: {OUTPUT_DIR!r}")
    os.makedirs(PLOT_DIR, exist_ok=True)
    raw = load_results()
    plot(raw)
