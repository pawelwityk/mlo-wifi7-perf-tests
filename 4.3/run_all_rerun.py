#!/usr/bin/env python3
"""
run_all_rerun.py  --  Scenario 4.3: Legacy Coexistence with IEEE 802.11be MLO

Based on [18]: D. Medda et al., "Investigating Inclusiveness and Backward
Compatibility of IEEE 802.11be Multi-link Operation",
2022 IEEE CSCN, pp. 20–24.

Sweeps over:
  - caseId:          'A', 'B', 'C'  (band assignment policy)
  - legacyFraction:  10 % to 90 % in 10 % steps  (as in paper Fig. 3–6)
  - RngRun:          1 … RUNS (default 20, paper: 150 repetitions)

Fixed parameters:
  - totalStas = 50
  - simTime   = 60 s (paper: 1 minute)
  - payloadSize = 1500 bytes (paper: 12000 bits)

Metrics collected:
  - AggThr_Legacy, AggThr_MLDs, AggThr_Total  [Mbit/s]
  - AvgThr_Legacy, AvgThr_MLDs  [Mbit/s]
  - JFI_Total, JFI_Legacy, JFI_MLDs

Persistence: per-run CSV files, resume-able after crash.
Plots produced (in plots/):
  4_3_agg_thr.svg    — Fig. 3 equivalent: aggregated total throughput vs legacy %
  4_3_agg_type.svg   — Fig. 4 equivalent: aggregated per-type throughput vs legacy %
  4_3_avg_type.svg   — Fig. 5 equivalent: average per-type throughput vs legacy %
  4_3_fairness.svg   — Fig. 6 equivalent: Jain's fairness index vs legacy %
"""

import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
NS3_BINARY   = "build/scratch/4.3/ns3.45-wifi-mlo-legacy-coexistence-optimized"
OUTPUT_DIR   = "thr_runs_4_3"
PLOT_DIR     = "plots"

CASES        = ["A", "B", "C"]
LEGACY_PCTS  = list(range(10, 100, 10))  # 10, 20, ..., 90  (%)
TOTAL_STAS   = 50
RUNS         = 20
SIM_TIME     = 60.0   # s
PAYLOAD      = 1500   # bytes

MAX_WORKERS  = 8
RERUN_MISSING = True

# Two-sided 95 % t critical values for df = 1..30
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365,  8: 2.306, 9: 2.262,10: 2.228,11: 2.201,12: 2.179,
   13: 2.160, 14: 2.145,15: 2.131,16: 2.120,17: 2.110,18: 2.101,
   19: 2.093, 20: 2.086,21: 2.080,22: 2.074,23: 2.069,24: 2.064,
   25: 2.060, 26: 2.056,27: 2.052,28: 2.048,29: 2.045,30: 2.042,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

# Regex patterns for output parsing
RE = {
    "agg_leg":  re.compile(r"AggThr_Legacy:\s*([0-9.]+)"),
    "agg_mld":  re.compile(r"AggThr_MLDs:\s*([0-9.]+)"),
    "agg_tot":  re.compile(r"AggThr_Total:\s*([0-9.]+)"),
    "avg_leg":  re.compile(r"AvgThr_Legacy:\s*([0-9.]+)"),
    "avg_mld":  re.compile(r"AvgThr_MLDs:\s*([0-9.]+)"),
    "jfi_tot":  re.compile(r"JFI_Total:\s*([0-9.]+)"),
    "jfi_leg":  re.compile(r"JFI_Legacy:\s*([0-9.]+)"),
    "jfi_mld":  re.compile(r"JFI_MLDs:\s*([0-9.]+)"),
}

FIELDS = ["agg_leg", "agg_mld", "agg_tot", "avg_leg", "avg_mld",
          "jfi_tot", "jfi_leg", "jfi_mld"]

CASE_COLORS  = {"A": "#E63B0E", "B": "#2E75D8", "C": "#28A745"}
CASE_MARKERS = {"A": "x",       "B": "+",       "C": "^"}
CASE_LABELS  = {"A": "Case A", "B": "Case B",   "C": "Case C"}


# ---------------------------------------------------------------------------
#  Persistence helpers
# ---------------------------------------------------------------------------

def _result_path(case: str, pct: int, rng: int) -> str:
    return os.path.join(OUTPUT_DIR, f"Case{case}_Pct{pct:02d}_rng{rng:03d}_result.csv")


def _parse_csv_row(text: str):
    """Return dict of field→float or None."""
    reader = csv.reader([text.strip()])
    parts  = [p.strip() for p in next(reader, []) if p.strip()]
    if len(parts) < len(FIELDS):
        return None
    try:
        return {k: float(v) for k, v in zip(FIELDS, parts)}
    except ValueError:
        return None


# ---------------------------------------------------------------------------
#  Run single job
# ---------------------------------------------------------------------------

def run_job(case: str, pct: int, rng: int, force: bool = False):
    fpath = _result_path(case, pct, rng)
    if os.path.exists(fpath) and not force:
        try:
            with open(fpath) as f:
                d = _parse_csv_row(f.read())
            if d is not None:
                return (case, pct, rng, d)
        except Exception:
            pass

    frac = pct / 100.0
    cmd  = [
        NS3_BINARY,
        f"--caseId={case}",
        f"--legacyFraction={frac}",
        f"--totalStas={TOTAL_STAS}",
        f"--simTime={SIM_TIME}",
        f"--payloadSize={PAYLOAD}",
        f"--RngRun={rng}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        parsed = {}
        for key, regex in RE.items():
            m = regex.search(result.stdout)
            if m:
                parsed[key] = float(m.group(1))
        if len(parsed) == len(FIELDS):
            with open(fpath, "w") as f:
                f.write(",".join(str(parsed[k]) for k in FIELDS) + "\n")
            return (case, pct, rng, parsed)
        else:
            print(f"  PARSE-FAIL Case{case} Pct{pct} rng{rng}", file=sys.stderr)
            print(f"  stdout: {result.stdout[-400:]}", file=sys.stderr)
    except Exception as e:
        print(f"  ERROR Case{case} Pct{pct} rng{rng}: {e}", file=sys.stderr)

    return (case, pct, rng, None)


# ---------------------------------------------------------------------------
#  Load all results
# ---------------------------------------------------------------------------

def load_results():
    """Return dict keyed by (case, pct) → list of field dicts."""
    data = {}
    if not os.path.isdir(OUTPUT_DIR):
        return data
    pat = re.compile(r"Case([ABC])_Pct(\d+)_rng(\d+)_result\.csv$")
    for fname in os.listdir(OUTPUT_DIR):
        m = pat.match(fname)
        if not m:
            continue
        case, pct, _rng = m.group(1), int(m.group(2)), int(m.group(3))
        if case not in CASES or pct not in LEGACY_PCTS:
            continue
        try:
            with open(os.path.join(OUTPUT_DIR, fname)) as f:
                d = _parse_csv_row(f.read())
            if d is None:
                continue
            data.setdefault((case, pct), []).append(d)
        except Exception:
            pass
    return data


# ---------------------------------------------------------------------------
#  CI helper
# ---------------------------------------------------------------------------

def ci95(vals):
    if len(vals) <= 1:
        return 0.0
    n = len(vals)
    t = T95.get(n - 1, 1.96)
    return t * stdev(vals) / (n ** 0.5)


def series(data, case, field):
    """Return (xs, means, cis) for a given case and field."""
    xs, ys, es = [], [], []
    for pct in LEGACY_PCTS:
        rows = data.get((case, pct), [])
        if not rows:
            continue
        vals = [r[field] for r in rows]
        xs.append(pct)
        ys.append(mean(vals))
        es.append(ci95(vals))
    return xs, ys, es


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------

def plot_all(data):
    # ── Fig 3: aggregated total throughput ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for case in CASES:
        xs, ys, es = series(data, case, "agg_tot")
        ax.errorbar(xs, ys, yerr=es, label=CASE_LABELS[case],
                    color=CASE_COLORS[case], marker=CASE_MARKERS[case],
                    capsize=3, linewidth=1.6)
    ax.set_xlabel("Percentage of legacy devices [%]")
    ax.set_ylabel("Aggregated system throughput [Mbit/s]")
    ax.set_title("Scenario 4.3: Aggregated Network-Wise Throughput")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "4_3_agg_thr.svg")
    fig.savefig(p, format="svg")
    print(f"[plot] {p}")
    plt.close(fig)

    # ── Fig 4: aggregated per-type throughput ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for case in CASES:
        for fld, label_sfx, ls in [("agg_leg", "Legacy", "-"), ("agg_mld", "MLDs", "--")]:
            xs, ys, es = series(data, case, fld)
            ax.errorbar(xs, ys, yerr=es,
                        label=f"{CASE_LABELS[case]} – {label_sfx}",
                        color=CASE_COLORS[case], marker=CASE_MARKERS[case],
                        linestyle=ls, capsize=3, linewidth=1.4)
    ax.set_xlabel("Percentage of legacy devices [%]")
    ax.set_ylabel("Aggregated throughput per device type [Mbit/s]")
    ax.set_title("Scenario 4.3: Aggregated Throughput per Device Type")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "4_3_agg_type.svg")
    fig.savefig(p, format="svg")
    print(f"[plot] {p}")
    plt.close(fig)

    # ── Fig 5: average per-type throughput ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for case in CASES:
        for fld, label_sfx, ls in [("avg_leg", "Legacy", "-"), ("avg_mld", "MLDs", "--")]:
            xs, ys, es = series(data, case, fld)
            ax.errorbar(xs, ys, yerr=es,
                        label=f"{CASE_LABELS[case]} – {label_sfx}",
                        color=CASE_COLORS[case], marker=CASE_MARKERS[case],
                        linestyle=ls, capsize=3, linewidth=1.4)
    ax.set_xlabel("Percentage of legacy devices [%]")
    ax.set_ylabel("Average throughput per device type [Mbit/s]")
    ax.set_title("Scenario 4.3: Average Throughput per Device Type")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "4_3_avg_type.svg")
    fig.savefig(p, format="svg")
    print(f"[plot] {p}")
    plt.close(fig)

    # ── Fig 6: Jain's Fairness Index ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # Left: total fairness
    for case in CASES:
        xs, ys, es = series(data, case, "jfi_tot")
        axes[0].errorbar(xs, ys, yerr=es, label=f"{CASE_LABELS[case]} Total",
                         color=CASE_COLORS[case], marker=CASE_MARKERS[case],
                         capsize=3, linewidth=1.6)
    axes[0].set_xlabel("Percentage of legacy devices [%]")
    axes[0].set_ylabel("Jain's fairness index")
    axes[0].set_title("Total Network Fairness")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    # Right: per-type fairness
    for case in CASES:
        for fld, label_sfx, ls in [("jfi_leg", "Legacy", "-"), ("jfi_mld", "MLDs", "--")]:
            xs, ys, es = series(data, case, fld)
            axes[1].errorbar(xs, ys, yerr=es,
                             label=f"{CASE_LABELS[case]} {label_sfx}",
                             color=CASE_COLORS[case], marker=CASE_MARKERS[case],
                             linestyle=ls, capsize=3, linewidth=1.4)
    axes[1].set_xlabel("Percentage of legacy devices [%]")
    axes[1].set_ylabel("Jain's fairness index")
    axes[1].set_title("Per-Type Fairness")
    axes[1].set_ylim(0.95, 1.01)
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Scenario 4.3: Fairness vs. Legacy Device Percentage")
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "4_3_fairness.svg")
    fig.savefig(p, format="svg")
    print(f"[plot] {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Main sweep
# ---------------------------------------------------------------------------

def main():
    print("[*] Loading existing results...", flush=True)
    existing = load_results()
    print(f"[*] Found {sum(len(v) for v in existing.values())} existing results", flush=True)

    jobs = list(product(CASES, LEGACY_PCTS, range(1, RUNS + 1)))
    print(f"[*] Total jobs: {len(jobs)} ({len(CASES)} cases × "
          f"{len(LEGACY_PCTS)} fractions × {RUNS} runs)", flush=True)

    failed = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_job, c, p, r): (c, p, r)
                   for c, p, r in jobs}
        for fut in as_completed(futures):
            completed += 1
            case, pct, rng, result = fut.result()
            if result is None:
                failed.append((case, pct, rng))
            if completed % 50 == 0:
                print(f"  [{completed}/{len(jobs)}] done", flush=True)

    print(f"[*] Parallel pass: {len(jobs)-len(failed)}/{len(jobs)} succeeded", flush=True)

    if RERUN_MISSING and failed:
        print(f"[*] Re-running {len(failed)} failed jobs...", flush=True)
        for i, (c, p, r) in enumerate(failed):
            run_job(c, p, r)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(failed)}] re-run done", flush=True)

    data = load_results()
    plot_all(data)
    print("[*] Done!", flush=True)


if __name__ == "__main__":
    main()
