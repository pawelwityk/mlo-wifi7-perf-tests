#!/usr/bin/env python3
"""Standalone plotter for 4.1.1: grouped bars (A-MPDU 64 vs 1024)."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401
import os, re, glob
import numpy as np
import matplotlib.pyplot as plt
from plot_style import ci95
from _robust import remove_outliers_robust as _ro

DATA_DIR = "thr_runs_4_1_1"
PLOT_FILE = "plots/4_1_1.svg"

SCENARIOS = ["1L  80 MHz", "1L 160 MHz", "2L STR  80 MHz", "2L STR 160 MHz"]
NMPDUS    = [64, 1024]

def safe(s): return s.replace(" ", "_")
PAT = re.compile(r"^(.+)_nm(\d+)_rng\d+_result\.csv$")

results = {n: {s: [] for s in SCENARIOS} for n in NMPDUS}
for fname in os.listdir(DATA_DIR):
    m = PAT.match(fname)
    if not m: continue
    safe_lbl, nm = m.group(1), int(m.group(2))
    label = next((s for s in SCENARIOS if safe(s) == safe_lbl), None)
    if label is None or nm not in NMPDUS: continue
    with open(os.path.join(DATA_DIR, fname)) as f:
        results[nm][label].append(float(f.read().strip()))

x = np.arange(len(SCENARIOS))
w = 0.27
means = {n: [np.mean(_ro(results[n][s])) if results[n][s] else 0 for s in SCENARIOS] for n in NMPDUS}
cis   = {n: [ci95(_ro(results[n][s]))    if results[n][s] else 0 for s in SCENARIOS] for n in NMPDUS}

fig, ax = plt.subplots(figsize=(10, 5))
offsets = {n: (i - (len(NMPDUS)-1)/2) * w for i, n in enumerate(NMPDUS)}
for n in NMPDUS:
    ax.bar(x + offsets[n], means[n], w, yerr=cis[n], capsize=4,
           label=f"A-MPDU = {n} MPDUs")
ax.set_xticks(x); ax.set_xticklabels(SCENARIOS, rotation=15)
ax.set_ylabel("Throughput [Mbit/s]")
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(PLOT_FILE)
print(f"[OK] {PLOT_FILE}")
for s in SCENARIOS:
    parts = [f"{n}={np.mean(results[n][s]) if results[n][s] else 0:8.2f} ±{ci95(results[n][s]) if results[n][s] else 0:.2f} (n={len(results[n][s])})"
             for n in NMPDUS]
    print(f"  {s:20s}  " + "  ".join(parts))
