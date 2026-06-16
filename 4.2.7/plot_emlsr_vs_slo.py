"""
Plotter for 4.2.7: EMLSR vs SLO — Throughput & Latency vs Normalized Load

Generates:
  - Throughput vs Load (one subplot per nStas)
  - Latency vs Load (one subplot per nStas)
Both with SLO and EMLSR curves, CI from 10 runs.
"""

# >>> thesis-style shim
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401
# <<< thesis-style shim

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# ===== CONFIG =====
OUTPUT_DIR = "thr_runs_4_2_7"
PLOT_DIR   = "plots"
RUNS       = 10
RNG_START  = 1001
CONFIDENCE = 0.95

MLO_MODES  = ["SLO", "EMLSR"]
NSTA_VALUES = [1, 4, 10]
NORMALIZED_LOADS = [round(0.1 * i, 1) for i in range(0, 11)]

LATENCY_LINTHRESH = 4.0  # symlog threshold

os.makedirs(PLOT_DIR, exist_ok=True)


def load_results():
    """Load all result CSVs into dict[mode][nStas][rho] -> list of (lat, thr)."""
    data = {m: {ns: {rho: [] for rho in NORMALIZED_LOADS}
                for ns in NSTA_VALUES} for m in MLO_MODES}

    pattern = re.compile(r"^M(SLO|EMLSR)_S(\d+)_R(\d+)_rng(\d+)_result\.csv$")
    if not os.path.isdir(OUTPUT_DIR):
        print(f"[!] Output directory '{OUTPUT_DIR}' not found.")
        return data

    for fname in os.listdir(OUTPUT_DIR):
        m = pattern.match(fname)
        if not m:
            continue
        mode = m.group(1)
        ns = int(m.group(2))
        rho = int(m.group(3)) / 10.0
        if mode not in MLO_MODES or ns not in NSTA_VALUES:
            continue
        if rho not in [round(0.1 * i, 1) for i in range(0, 11)]:
            continue
        try:
            with open(os.path.join(OUTPUT_DIR, fname)) as f:
                parts = f.read().strip().split(",")
                lat, thr = float(parts[0]), float(parts[1])
                data[mode][ns][rho].append((lat, thr))
        except (ValueError, IndexError):
            pass
    return data


def ci_mean(values, confidence=CONFIDENCE):
    """Return (mean, ci_half_width)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = np.mean(values)
    if n == 1:
        return mean, 0.0
    se = sp_stats.sem(values)
    t_crit = sp_stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, t_crit * se


def plot_throughput(data):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("EMLSR vs SLO — DL Throughput vs Normalized Load", fontsize=13)

    colors = {"SLO": "tab:blue", "EMLSR": "tab:red"}
    markers = {"SLO": "o", "EMLSR": "s"}

    for idx, ns in enumerate(NSTA_VALUES):
        ax = axes[idx]
        for mode in MLO_MODES:
            means, cis, xs = [], [], []
            for rho in NORMALIZED_LOADS:
                vals = [t for _, t in data[mode][ns][rho]]
                if not vals:
                    continue
                m, c = ci_mean(vals)
                means.append(m)
                cis.append(c)
                xs.append(rho)
            if xs:
                ax.errorbar(xs, means, yerr=cis, label=mode,
                            color=colors[mode], marker=markers[mode],
                            markersize=5, capsize=3, linewidth=1.5)
        ax.set_xlabel("Normalized Load")
        ax.set_title(f"{ns} STA{'s' if ns > 1 else ''}")
        ax.set_xlim(-0.05, 1.05)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean DL Throughput [Mbit/s]")
    plt.tight_layout()
    base = os.path.join(PLOT_DIR, "4_2_7_throughput_vs_load")
    fig.savefig(base + ".svg", format="svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {base}.{{svg,pdf}}")
    plt.close(fig)


def plot_latency(data):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("EMLSR vs SLO — DL Latency vs Normalized Load", fontsize=13)

    colors = {"SLO": "tab:blue", "EMLSR": "tab:red"}
    markers = {"SLO": "o", "EMLSR": "s"}

    for idx, ns in enumerate(NSTA_VALUES):
        ax = axes[idx]
        for mode in MLO_MODES:
            means, cis, xs = [], [], []
            for rho in NORMALIZED_LOADS:
                vals = [la for la, _ in data[mode][ns][rho]]
                if not vals:
                    continue
                m, c = ci_mean(vals)
                means.append(m)
                cis.append(c)
                xs.append(rho)
            if xs:
                ax.errorbar(xs, means, yerr=cis, label=mode,
                            color=colors[mode], marker=markers[mode],
                            markersize=5, capsize=3, linewidth=1.5)
        ax.set_xlabel("Normalized Load")
        ax.set_title(f"{ns} STA{'s' if ns > 1 else ''}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_yscale("symlog", linthresh=LATENCY_LINTHRESH, linscale=1.0)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Channel access delay [ms]")
    plt.tight_layout()
    base = os.path.join(PLOT_DIR, "4_2_7_latency_vs_load")
    fig.savefig(base + ".svg", format="svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {base}.{{svg,pdf}}")
    plt.close(fig)


def main():
    data = load_results()
    plot_throughput(data)
    plot_latency(data)


if __name__ == "__main__":
    main()
