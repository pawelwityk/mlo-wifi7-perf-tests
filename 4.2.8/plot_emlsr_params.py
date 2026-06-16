"""
Plotter for 4.2.8: EMLSR Parameter Variation — Throughput & Latency

Generates one pair of plots (throughput + latency) per swept parameter,
with the parameter value on x-axis and separate curves for each nStas.
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
OUTPUT_DIR = "thr_runs_4_2_8"
PLOT_DIR   = "plots"
RUNS       = 10
RNG_START  = 1001
CONFIDENCE = 0.95

NSTA_VALUES = [1, 4]

SWEEPS = {
    "transDelay": {"values": [0, 16, 32, 64, 128, 256], "xlabel": "Transition Delay [µs]"},
    "padDelay":   {"values": [0, 32, 64, 128, 256], "xlabel": "Padding Delay [µs]"},
    "timeout":    {"values": [128, 256, 512, 1024, 2048, 4096, 8192, 16384], "xlabel": "Transition Timeout [µs]"},
    "auxWidth":   {"values": [20, 40, 80], "xlabel": "Aux PHY Width [MHz]"},
    "sleep":      {"values": [0, 1], "xlabel": "Put Aux PHY to Sleep"},
    "txCap":      {"values": [0, 1], "xlabel": "Aux PHY TX Capable"},
    "switchAux":  {"values": [0, 1], "xlabel": "Switch Aux PHY"},
}

LATENCY_LINTHRESH = 4.0  # symlog threshold

os.makedirs(PLOT_DIR, exist_ok=True)


def load_results():
    """Load all results into dict[param][val][nStas] -> list of (lat, thr)."""
    data = {}
    for param, info in SWEEPS.items():
        data[param] = {v: {ns: [] for ns in NSTA_VALUES} for v in info["values"]}

    pattern = re.compile(r"^P(\w+)_V([^_]+)_S(\d+)_rng(\d+)_result\.csv$")
    if not os.path.isdir(OUTPUT_DIR):
        print(f"[!] Output directory '{OUTPUT_DIR}' not found.")
        return data

    for fname in os.listdir(OUTPUT_DIR):
        m = pattern.match(fname)
        if not m:
            continue
        param, val_s, ns_s = m.group(1), m.group(2), int(m.group(3))
        if param not in data:
            continue
        try:
            val = int(val_s)
        except ValueError:
            val = float(val_s)
        if val not in data[param] or ns_s not in NSTA_VALUES:
            continue
        try:
            with open(os.path.join(OUTPUT_DIR, fname)) as f:
                parts = f.read().strip().split(",")
                lat, thr = float(parts[0]), float(parts[1])
                data[param][val][ns_s].append((lat, thr))
        except (ValueError, IndexError):
            pass
    return data


def ci_mean(values, confidence=CONFIDENCE):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = np.mean(values)
    if n == 1:
        return mean, 0.0
    se = sp_stats.sem(values)
    t_crit = sp_stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, t_crit * se


def plot_parameter(data, param, info):
    """Create throughput + latency plot for one parameter."""
    values = info["values"]
    xlabel = info["xlabel"]

    colors = {1: "tab:blue", 4: "tab:red"}
    markers = {1: "o", 4: "s"}

    # --- Throughput ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ns in NSTA_VALUES:
        means, cis, xs = [], [], []
        for v in values:
            vals = [t for _, t in data[param][v][ns]]
            if not vals:
                continue
            m, c = ci_mean(vals)
            means.append(m)
            cis.append(c)
            xs.append(v)
        if xs:
            ax.errorbar(xs, means, yerr=cis,
                        label=f"{ns} STA{'s' if ns > 1 else ''}",
                        color=colors[ns], marker=markers[ns],
                        markersize=6, capsize=3, linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean DL Throughput [Mbit/s]")
    ax.set_title(f"EMLSR — Throughput vs {xlabel}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if param == "timeout":
        ax.set_xscale("log", base=2)
    plt.tight_layout()
    base = os.path.join(PLOT_DIR, f"4_2_8_{param}_throughput")
    fig.savefig(base + ".svg", format="svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {base}.{{svg,pdf}}")
    plt.close(fig)

    # --- Latency ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ns in NSTA_VALUES:
        means, cis, xs = [], [], []
        for v in values:
            vals = [la for la, _ in data[param][v][ns]]
            if not vals:
                continue
            m, c = ci_mean(vals)
            means.append(m)
            cis.append(c)
            xs.append(v)
        if xs:
            ax.errorbar(xs, means, yerr=cis,
                        label=f"{ns} STA{'s' if ns > 1 else ''}",
                        color=colors[ns], marker=markers[ns],
                        markersize=6, capsize=3, linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Channel access delay [ms]")
    ax.set_title(f"EMLSR — Channel Access Delay vs {xlabel}")
    ax.set_yscale("symlog", linthresh=LATENCY_LINTHRESH, linscale=1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if param == "timeout":
        ax.set_xscale("log", base=2)
    plt.tight_layout()
    base = os.path.join(PLOT_DIR, f"4_2_8_{param}_latency")
    fig.savefig(base + ".svg", format="svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {base}.{{svg,pdf}}")
    plt.close(fig)


def main():
    data = load_results()
    for param, info in SWEEPS.items():
        plot_parameter(data, param, info)


if __name__ == "__main__":
    main()
