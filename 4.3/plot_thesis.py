#!/usr/bin/env python3
"""
Scenario 4.3 — Thesis-faithful plotter for Figs 4.14, 4.15, 4.16
of praca_mgr (J. Koziol, "Performance Analysis of MLO in IEEE 802.11 Networks").

Reuses results produced by ``scratch/4.3/run_all_rerun.py`` (writes to
``thr_runs_4_3/``).  Maps simulator's caseId → MLO mode so labels match
the thesis:
    Case A  →  "EMLSR + legacy"   (single active MLD link, AP-lock approx)
    Case B  →  "STR + legacy"     (3-link MLDs, fully sharing 2.4 GHz)

Produces the three subplots described in §4.3 of the thesis:
    plots/4_14.svg + .pdf   — Network throughput vs % legacy
    plots/4_15.svg + .pdf   — Throughput per device type vs % legacy
    plots/4_16.svg + .pdf   — Average throughput per device type vs % legacy
"""
from __future__ import annotations

import os
import re
import sys
from statistics import mean

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scratch", "_common"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot_style  # noqa: E402
from _robust import ci_hw_robust, remove_outliers_robust  # noqa: E402

plot_style.install()

# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(ROOT, "thr_runs_4_3")
LEGACY_PCTS = list(range(10, 100, 10))

# Thesis-style mode mapping  (caseId  →  display label, color)
MODE_MAP = {
    "A": ("EMLSR + legacy", "#2E75D8"),
    "B": ("STR + legacy", "#E63B0E"),
}

FIELDS = ["agg_leg", "agg_mld", "agg_tot", "avg_leg", "avg_mld",
          "jfi_tot", "jfi_leg", "jfi_mld"]


def _load() -> dict:
    """Return {(case, pct) -> {field -> [values]}}."""
    out: dict = {}
    if not os.path.isdir(OUTPUT_DIR):
        return out
    pat = re.compile(r"Case([AB])_Pct(\d+)_rng\d+_result\.csv$")
    for fname in os.listdir(OUTPUT_DIR):
        m = pat.match(fname)
        if not m:
            continue
        case, pct = m.group(1), int(m.group(2))
        if pct not in LEGACY_PCTS:
            continue
        try:
            with open(os.path.join(OUTPUT_DIR, fname)) as f:
                parts = [p.strip() for p in f.read().strip().split(",")]
            if len(parts) < len(FIELDS):
                continue
            row = {k: float(v) for k, v in zip(FIELDS, parts)}
        except Exception:
            continue
        bucket = out.setdefault((case, pct), {f: [] for f in FIELDS})
        for f in FIELDS:
            bucket[f].append(row[f])
    return out


def _series(data: dict, case: str, field: str):
    """Return (xs, ys, errs) using mean and *standard error of the mean* (SEM)
    after IQR/MAD outlier filtering. SEM is used (not 95% CI t-half-width)
    to avoid the huge whiskers that crowd out the signal in plots 4.14–4.16.

    Buckets with fewer than 3 valid samples after cleaning are skipped to
    prevent isolated outliers (e.g. the spurious first point of the legacy
    series in 4.16) from distorting the curve.
    """
    import numpy as _np
    xs, ys, es = [], [], []
    for pct in LEGACY_PCTS:
        bucket = data.get((case, pct))
        if not bucket or not bucket[field]:
            continue
        vals = remove_outliers_robust(bucket[field])
        if len(vals) < 3:
            continue
        arr = _np.asarray(vals, dtype=float)
        m = float(arr.mean())
        sem = float(arr.std(ddof=1) / _np.sqrt(len(arr)))
        xs.append(pct)
        ys.append(m)
        es.append(sem)
    return xs, ys, es


def _save(fig, name: str) -> None:
    out = os.path.join(plot_style.PLOT_DIR, name)
    fig.savefig(out + ".svg", format="svg")
    fig.savefig(out + ".pdf", format="pdf")
    print(f"[plot] {out}.{{svg,pdf}}")
    plt.close(fig)


def plot_4_14(data: dict) -> None:
    """Fig 4.14 — Network throughput vs percentage of legacy stations."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    all_y = []
    for case, (label, color) in MODE_MAP.items():
        xs, ys, es = _series(data, case, "agg_tot")
        if not xs:
            continue
        all_y.extend(ys)
        ax.errorbar(xs, ys, yerr=es, label=label, color=color,
                    marker="o", linewidth=1.6, capsize=2.0)
    ax.set_xlabel("Percentage of legacy devices")
    ax.set_ylabel("Aggregated system throughput [Mbit/s]")
    ax.set_xticks(LEGACY_PCTS)
    # Extra headroom above the highest data point so the top-left legend
    # has room to sit clear of the curves.
    if all_y:
        ax.set_ylim(0, max(all_y) * 1.30)
    else:
        ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "4_14")


def plot_4_15(data: dict) -> None:
    """Fig 4.15 — Aggregated throughput per device type vs % legacy."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    all_y = []
    for case, (label, color) in MODE_MAP.items():
        for fld, sfx, ls in [("agg_mld", "MLDs", "-"), ("agg_leg", "Legacy", "--")]:
            xs, ys, es = _series(data, case, fld)
            if not xs:
                continue
            all_y.extend(ys)
            ax.errorbar(xs, ys, yerr=es,
                        label=f"{label} \u2013 {sfx}",
                        color=color, marker="o", linestyle=ls,
                        linewidth=1.4, capsize=2.0)
    ax.set_xlabel("Percentage of legacy devices")
    ax.set_ylabel("Aggregated throughput per device type [Mbit/s]")
    ax.set_xticks(LEGACY_PCTS)
    if all_y:
        ax.set_ylim(0, max(all_y) * 1.45)
    else:
        ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "4_15")


def plot_4_16(data: dict) -> None:
    """Fig 4.16 — Average throughput per device type vs % legacy."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for case, (label, color) in MODE_MAP.items():
        for fld, sfx, ls in [("avg_mld", "MLDs", "-"), ("avg_leg", "Legacy", "--")]:
            xs, ys, es = _series(data, case, fld)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es,
                        label=f"{label} \u2013 {sfx}",
                        color=color, marker="o", linestyle=ls,
                        linewidth=1.4, capsize=2.0)
    ax.set_xlabel("Percentage of legacy devices")
    ax.set_ylabel("Average throughput per device type [Mbit/s]")
    ax.set_xticks(LEGACY_PCTS)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "4_16")


def main() -> int:
    data = _load()
    n_pts = sum(len(v.get("agg_tot", [])) for v in data.values())
    print(f"[load] {len(data)} (case,pct) buckets, {n_pts} samples total")
    if n_pts == 0:
        print("[ERROR] No data in thr_runs_4_3/.  Run the simulator first:", file=sys.stderr)
        print("        python3 scratch/4.3/run_all_rerun.py", file=sys.stderr)
        return 1
    plot_4_14(data)
    plot_4_15(data)
    plot_4_16(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
