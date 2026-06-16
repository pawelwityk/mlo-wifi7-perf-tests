#!/usr/bin/env python3
"""
Scenario 4.5 — Comparison of Results (thesis Figs 4.18, 4.19, 4.20, 4.21).

Pure post-processor: aggregates already-existing per-scenario CSV results in
``thr_runs_4_*/`` directories produced by 4.1.* / 4.2.* sweeps, computes
percentage gains/reductions/increases between MLO and SLO, and emits the four
summary bar charts.

    plots/4_18.svg + .pdf  — Throughput gain of MLO modes over SLO
    plots/4_19.svg + .pdf  — Latency reduction MLO vs SLO (contention-free)
    plots/4_20.svg + .pdf  — Latency increase MLO vs SLO (contention)
    plots/4_21.svg + .pdf  — Jain's Fairness Index in contention scenarios

Each summary number is computed from the underlying robust mean (IQR + MAD
filtered) of the matching per-scenario observations.  Where a particular
metric/configuration is unavailable, its bar is omitted with a warning.
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from statistics import mean

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scratch", "_common"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_style  # noqa: E402
from _robust import remove_outliers_robust  # noqa: E402

plot_style.install()

THR = {
    "4.1.1": os.path.join(ROOT, "thr_runs_4_1_1"),  # Baseline
    "4.1.3": os.path.join(ROOT, "thr_runs_4_1_3"),  # OBSS load
    "4.1.4": os.path.join(ROOT, "thr_runs_4_1_4"),  # MPDU agg / contention
    "4.1.5": os.path.join(ROOT, "thr_runs_4_1_5"),  # MPDU agg / 10% occ
    "4.2.1": os.path.join(ROOT, "thr_runs_4_2_1"),  # Latency #1
    "4.2.2": os.path.join(ROOT, "thr_runs_4_2_2_precise"),  # Latency #2
    "4.2.3": os.path.join(ROOT, "thr_runs_4_2_3"),  # Latency #3
    "4.2.4": os.path.join(ROOT, "thr_runs_4_2_4"),  # Latency #4A
    "4.2.5": os.path.join(ROOT, "thr_runs_4_2_5"),  # Latency #4B
    "4.2.6": os.path.join(ROOT, "thr_runs_4_2_6"),  # Latency #5
}

COLOR_GAIN = "#28A745"   # green for gain
COLOR_RED  = "#E63B0E"   # red for increase
COLOR_BLUE = "#2E75D8"


def _save(fig, name: str) -> None:
    out = os.path.join(plot_style.PLOT_DIR, name)
    fig.savefig(out + ".svg", format="svg")
    fig.savefig(out + ".pdf", format="pdf")
    print(f"[plot] {out}.{{svg,pdf}}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Scenario-specific loaders
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _load_csv_first_col(path: str) -> float | None:
    try:
        with open(path) as f:
            txt = f.read().strip()
        if not txt:
            return None
        v = txt.split(",")[0].strip()
        return float(v)
    except Exception:
        return None


def _load_csv_cols(path: str) -> list[float] | None:
    try:
        with open(path) as f:
            txt = f.read().strip()
        if not txt:
            return None
        return [float(x.strip()) for x in txt.split(",") if x.strip()]
    except Exception:
        return None


def _safe_mean(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan")
    vals = remove_outliers_robust(vals)
    if not vals:
        return float("nan")
    return float(mean(vals))


def load_4_1_1() -> dict:
    """{(nLinks, agg) -> mean throughput}.  Used for Fig 4.18 (Baseline)."""
    d = defaultdict(list)
    folder = THR["4.1.1"]
    if not os.path.isdir(folder):
        return {}
    # Filename format: "<nL>L[_STR]_<bw>_MHz_nm<nm>_rng<rng>_result.csv".
    # 80 MHz files have a double underscore (e.g. "2L_STR__80_MHz_…") so the
    # regex must accept one-or-more underscores around the optional STR_ tag.
    pat = re.compile(r"^(\d)L_+(?:STR_+)?(\d+)_MHz_nm(\d+)_rng\d+_result\.csv$")
    for fn in os.listdir(folder):
        m = pat.match(fn)
        if not m:
            continue
        nl, _bw, nm = int(m.group(1)), int(m.group(2)), int(m.group(3))
        v = _load_csv_first_col(os.path.join(folder, fn))
        if v is not None:
            d[(nl, nm)].append(v)
    return {k: _safe_mean(v) for k, v in d.items()}


def load_4_1_4_obss() -> dict:
    """{mode -> mean throughput} averaged over the *whole* OBSS-density sweep
    (n>=1, i.e. excluding the no-contention baseline n=0).  This mirrors
    praca_mgr's Fig 4.18 methodology, which compares average performance over
    a range of OBSS conditions rather than the single worst case.

    Two-stage average: first per-n mean (so heavy collapse at high n doesn't
    dominate the robust outlier filter), then unweighted mean across n
    buckets.  Without this, ``_safe_mean``'s outlier removal trims the
    high-throughput n=1..3 samples as outliers from the bimodal distribution,
    artificially inflating the MLO/SLO ratio.
    """
    folder = THR["4.1.4"]
    if not os.path.isdir(folder):
        return {}
    # New per-direction file format: mode_direction_n_rng_result.csv with a
    # single throughput value. Average DL and UL together per (mode, n) bucket
    # so the resulting "mode mean" reflects both directions, mirroring how
    # praca_mgr Fig 4.18 aggregates across the sweep.
    bucket = defaultdict(lambda: defaultdict(list))  # mode -> n -> [v]
    pat = re.compile(r"^(SLO|EMLSR|STR)_(DL|UL)_n(\d+)_rng\d+_result\.csv$")
    for fn in os.listdir(folder):
        m = pat.match(fn)
        if not m:
            continue
        mode, _direction, n = m.group(1), m.group(2), int(m.group(3))
        if n == 0:
            continue   # exclude no-contention baseline
        cols = _load_csv_cols(os.path.join(folder, fn))
        if cols and len(cols) >= 1:
            bucket[mode][n].append(cols[0])
    out = {}
    for mode, by_n in bucket.items():
        per_n = [mean(vs) for vs in by_n.values() if vs]
        if per_n:
            out[mode] = float(mean(per_n))
    return out


# Per-folder schema: index of the *mean latency* column in the per-run CSV.
# (All 4.2.* CSVs put mean_lat in column 0; the legacy code's `min(cols)`
# was incorrectly returning the p50 column for several scenarios.)
_MEAN_LAT_COL = {
    "4.2.1": 0,   # lat, throughput
    "4.2.2": 0,   # mean_lat, p50, p99, throughput
    "4.2.3": 0,   # mean_lat, p50, p99, throughput
    "4.2.4": 0,   # mean_lat, p50, p95, p99, throughput
    "4.2.5": 0,   # mean_lat, p50, p95, p99, throughput
    "4.2.6": 0,   # mean_lat, p50, p95, p99, throughput
}


def _classify_mode(folder_key: str, fn: str) -> str | None:
    """Map a per-run CSV filename to a logical mode label.

    Folder-aware so that e.g. 4.2.5's MEMLSR2/MSTR2/MSLO/MSTR1P1/MSTR5
    naming is decoded into stable {SLO, STR, EMLSR, STR1P1} buckets.
    """
    upper = fn.upper()
    if folder_key == "4.2.1":
        # L{n}_W{bw}_S{sta}_R{rho}_rng* — 1L = SLO, 2L = STR
        m = re.match(r"L(\d)_", upper)
        if m:
            return {1: "SLO", 2: "STR"}.get(int(m.group(1)))
        return None
    if folder_key == "4.2.2":
        m = re.match(r"L(\d)_", upper)
        if m:
            return {1: "SLO", 2: "STR", 4: "STR"}.get(int(m.group(1)))
        return None
    if folder_key == "4.2.3":
        # L{n}_M{STR|EMLSR}_Load*
        m = re.match(r"L(\d)_M(STR|EMLSR)_", upper)
        if not m:
            return None
        nl, mode = int(m.group(1)), m.group(2)
        if nl == 1:
            return "SLO"
        return mode
    if folder_key == "4.2.4":
        # L{n}_Load* — 1L=SLO, 2L=STR2, 4L=STR4 → bucket as SLO / STR
        m = re.match(r"L(\d)_", upper)
        if m:
            return {1: "SLO", 2: "STR", 4: "STR"}.get(int(m.group(1)))
        return None
    if folder_key == "4.2.5":
        # MSLO, MSTR2, MEMLSR2, MSTR1P1, MSTR5
        m = re.match(r"M([A-Z0-9]+)_LOAD", upper)
        if not m:
            return None
        tag = m.group(1)
        return {
            "SLO":   "SLO",
            "STR2":  "STR",
            "EMLSR2": "EMLSR",
            "STR1P1": "STR1P1",
            "STR5":  "STR",
        }.get(tag)
    if folder_key == "4.2.6":
        # NL{1|2}_N1{a}_N2{b}_Load* — NL1=SLO, NL2=STR
        m = re.match(r"NL(\d)_", upper)
        if m:
            return {1: "SLO", 2: "STR"}.get(int(m.group(1)))
        return None
    return None


def load_4_2_lat(folder_key: str) -> dict:
    """Mean latency per mode for a 4.2.* scenario, aggregated across all
    sweep points (loads, OBSS configs, etc.) using robust mean.

    Returns ``{mode -> mean_ms}``; missing modes simply do not appear.
    """
    folder = THR.get(folder_key)
    if not folder or not os.path.isdir(folder):
        return {}
    col_ix = _MEAN_LAT_COL.get(folder_key, 0)
    out = defaultdict(list)
    for fn in os.listdir(folder):
        mode = _classify_mode(folder_key, fn)
        if mode is None:
            continue
        cols = _load_csv_cols(os.path.join(folder, fn))
        if not cols or col_ix >= len(cols):
            continue
        v = cols[col_ix]
        if v is None or not (0.0 < v < 1.0e4):
            continue
        out[mode].append(v)
    return {k: _safe_mean(v) for k, v in out.items() if v}


# ---------------------------------------------------------------------------
#  Plotters
# ---------------------------------------------------------------------------

def _agg_thr_4_1_3(mode: str) -> float | None:
    """Mean throughput per mode for Throughput Scenario 3 (4.1.3).

    Files are ``{STR|EMLSR}_rng*_result.csv`` with three columns:
    ``BSS0,BSS1,BSS2`` [Mbit/s].  BSS0 and BSS1 are *always* SLO baselines;
    only BSS2 switches between STR/EMLSR.  The SLO neighbors' throughput
    depends on which mode BSS2 is in (aggressive STR squeezes them harder
    than EMLSR), so the SLO baseline is computed per BSS2 mode:
      * mode == "SLO_STR"   -> mean of col0+col1 from STR_rng*.csv
      * mode == "SLO_EMLSR" -> mean of col0+col1 from EMLSR_rng*.csv
      * mode == "STR"       -> col2 of STR_rng*.csv
      * mode == "EMLSR"     -> col2 of EMLSR_rng*.csv
    """
    folder = THR["4.1.3"]
    if not os.path.isdir(folder):
        return None
    pat = re.compile(r"^(STR|EMLSR)_rng\d+_result\.csv$")
    vals = []
    for fn in os.listdir(folder):
        m = pat.match(fn)
        if not m:
            continue
        cols = _load_csv_cols(os.path.join(folder, fn))
        if cols is None or len(cols) < 3:
            continue
        run_mode = m.group(1)
        if mode == f"SLO_{run_mode}":
            vals.extend([cols[0], cols[1]])
        elif mode == run_mode:
            vals.append(cols[2])
    if not vals:
        return None
    return _safe_mean(vals)


def plot_4_18(thr_baseline: dict, thr_obss: dict) -> None:
    """Fig 4.18 — Throughput gain of MLO over SLO. Two sub-plots in one PDF:
    (a) MLO-STR  → bars: Baseline / Throughput Eval #1 / Throughput Eval #3
    (b) MLO-EMLSR → bars: Throughput Eval #1 / Throughput Eval #3
    Mirrors praca_mgr Fig 4.18 (a) + (b).
    """
    # ---- (a) MLO-STR -------------------------------------------------------
    str_bars = []
    slo_b = thr_baseline.get((1, 1024))
    str_b = thr_baseline.get((2, 1024))
    if slo_b and str_b:
        str_bars.append(("Throughput\nScenario 1",
                         100.0 * (str_b - slo_b) / slo_b))
    slo_str_e2 = _agg_thr_4_1_3("SLO_STR")
    str_e2     = _agg_thr_4_1_3("STR")
    if slo_str_e2 and str_e2:
        str_bars.append(("Throughput\nScenario 3",
                         100.0 * (str_e2 - slo_str_e2) / slo_str_e2))
    slo_o = thr_obss.get("SLO"); str_o = thr_obss.get("STR")
    if slo_o and str_o:
        str_bars.append(("Throughput\nScenario 4",
                         100.0 * (str_o - slo_o) / slo_o))

    # ---- (b) MLO-EMLSR -----------------------------------------------------
    em_bars = []
    slo_em_e2 = _agg_thr_4_1_3("SLO_EMLSR")
    emlsr_e2  = _agg_thr_4_1_3("EMLSR")
    if slo_em_e2 and emlsr_e2:
        em_bars.append(("Throughput\nScenario 3",
                        100.0 * (emlsr_e2 - slo_em_e2) / slo_em_e2))
    emlsr_o = thr_obss.get("EMLSR")
    if slo_o and emlsr_o:
        em_bars.append(("Throughput\nScenario 4",
                        100.0 * (emlsr_o - slo_o) / slo_o))

    if not str_bars and not em_bars:
        print("[skip] 4.18 no data", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = [("(a) MLO-STR", str_bars, "#3b1f70"),
              ("(b) MLO-EMLSR", em_bars, "#2A9D8F")]
    for ax, (sub, bars, color) in zip(axes, panels):
        if not bars:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(sub, fontsize=10)
            continue
        labels = [b[0] for b in bars]
        vals = [b[1] for b in bars]
        xs = np.arange(len(bars))
        ax.bar(xs, vals, color=color, width=0.55,
               edgecolor="black", linewidth=0.6)
        ymax = max(vals) if max(vals) > 0 else 1.0
        for x, v in zip(xs, vals):
            ax.text(x, v + ymax * 0.02, f"{v:.2f}%",
                    ha="center", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel(sub, fontsize=10)
        ax.set_ylabel("Throughput gain [%]", fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(min(0, min(vals) * 1.18), ymax * 1.28)
        ax.axhline(0, color="black", linewidth=0.5)
    fig.suptitle("Fig. 4.18 — Throughput gain of MLO modes over SLO",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "4_18")


def _stack_lat_ratio(data: dict, label_for: dict, sign: str = "reduce"):
    """For each scenario in `data` produce (label, pct_change) where SLO is
    baseline.  sign='reduce' returns (slo-mlo)/slo*100;  sign='increase' the
    opposite.
    """
    rows = []
    for scen, modes in data.items():
        slo = modes.get("SLO")
        if not slo or slo <= 0:
            continue
        for mode in ("STR", "EMLSR"):
            v = modes.get(mode)
            if not v:
                continue
            if sign == "reduce":
                pct = 100.0 * (slo - v) / slo
            else:
                pct = 100.0 * (v - slo) / slo
            rows.append((f"{label_for[scen]}\n{mode}", pct))
    return rows


def plot_4_19() -> None:
    """Fig 4.19 — Latency change MLO vs SLO (contention-free: 4.2.1, 4.2.2,
    4.2.3). Signed change: negative = MLO faster (less is better).
    """
    scenarios = [("Latency\nScenario 1", "4.2.1"),
                 ("Latency\nScenario 2", "4.2.2"),
                 ("Latency\nScenario 3", "4.2.3")]
    rows = []
    for label, key in scenarios:
        modes = load_4_2_lat(key)
        slo, mlo = modes.get("SLO"), modes.get("STR")
        if not slo or not mlo or slo <= 0:
            continue
        pct = 100.0 * (mlo - slo) / slo
        rows.append((label, pct))
    if not rows:
        print("[skip] 4.19 no data", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [("#2A9D8F" if v < 0 else "#3b1f70") for v in vals]
    xs = np.arange(len(rows))
    ax.bar(xs, vals, color=colors, width=0.55,
           edgecolor="black", linewidth=0.6)
    span = max(vals) - min(vals) if vals else 1.0
    if span <= 0:
        span = 1.0
    for x, v in zip(xs, vals):
        offset = span * 0.02
        ax.text(x, v + (offset if v >= 0 else -offset),
                f"{v:+.2f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Latency change [%]")
    ax.set_xlabel("Scenarios")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    if vals:
        vmin = min(min(vals), 0.0)
        vmax = max(max(vals), 0.0)
        pad = max(span, abs(vmax - vmin)) * 0.18
        ax.set_ylim(vmin - pad, vmax + pad)
    # "less is better" annotation along the Y-axis.
    ax.annotate("less is better ↓",
                xy=(0.01, 0.98), xycoords="axes fraction",
                ha="left", va="top",
                fontsize=10, fontstyle="italic", color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#bbbbbb", lw=0.6))
    ax.set_title("Fig. 4.19 — Latency change MLO vs SLO (contention-free)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "4_19")


def plot_4_20() -> None:
    """Fig 4.20 — Latency change MLO vs SLO (contention scenarios).

    Bars (mirroring praca_mgr Fig 4.20):
      - Latency Evaluation #4A             : 4.2.4  STR vs SLO
      - Latency Evaluation #4B (MLO-EMLSR) : 4.2.5  EMLSR vs SLO
      - Latency Evaluation #4B (MLO-STR1+1): 4.2.5  STR1P1 vs SLO
      - Latency Evaluation #5              : 4.2.6  STR vs SLO
    """
    bars = []
    m4a = load_4_2_lat("4.2.4")
    if m4a.get("SLO") and m4a.get("STR"):
        bars.append(("Latency Scenario 4",
                     100.0 * (m4a["STR"] - m4a["SLO"]) / m4a["SLO"]))
    m4b = load_4_2_lat("4.2.5")
    if m4b.get("SLO") and m4b.get("EMLSR"):
        bars.append(("Latency Scenario 5\n(MLO-EMLSR)",
                     100.0 * (m4b["EMLSR"] - m4b["SLO"]) / m4b["SLO"]))
    if m4b.get("SLO") and m4b.get("STR1P1"):
        bars.append(("Latency Scenario 5\n(MLO-STR1+1)",
                     100.0 * (m4b["STR1P1"] - m4b["SLO"]) / m4b["SLO"]))
    m5 = load_4_2_lat("4.2.6")
    if m5.get("SLO") and m5.get("STR"):
        bars.append(("Latency Scenario 6",
                     100.0 * (m5["STR"] - m5["SLO"]) / m5["SLO"]))

    if not bars:
        print("[skip] 4.20 no data", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = [("#2A9D8F" if v < 0 else "#3b1f70") for v in vals]
    xs = np.arange(len(bars))
    ax.bar(xs, vals, color=colors, width=0.55,
           edgecolor="black", linewidth=0.6)
    span = max(vals) - min(vals) if vals else 1.0
    if span <= 0:
        span = 1.0
    for x, v in zip(xs, vals):
        offset = span * 0.02
        ax.text(x, v + (offset if v >= 0 else -offset),
                f"{v:+.2f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Scenarios")
    ax.set_ylabel("Latency change [%]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    # Extra headroom so above/below value labels are not clipped by the frame
    if vals:
        vmin = min(min(vals), 0.0)
        vmax = max(max(vals), 0.0)
        pad = max(span, abs(vmax - vmin)) * 0.18
        ax.set_ylim(vmin - pad, vmax + pad)
    # "less is better" annotation along the Y-axis.
    ax.annotate("less is better ↓",
                xy=(0.01, 0.98), xycoords="axes fraction",
                ha="left", va="top",
                fontsize=10, fontstyle="italic", color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#bbbbbb", lw=0.6))
    ax.set_title("Fig. 4.20 — Latency change MLO vs SLO (contention)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "4_20")


def plot_4_21() -> None:
    """Fig 4.21 — Jain's Fairness Index in throughput contention scenarios.

    Mirrors praca_mgr Fig 4.21 structure:
      - Throughput Scenario 3 (EMLSR / STR)   → JFI across BSS0+BSS1+BSS2 from 4.1.3
      - MLO and Legacy Stations (EMLSR / STR) → JFI from 4.3 Cases A and B
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    rows = []
    colors = []

    # ── Throughput Scenario 3 (4.1.3): JFI across the 3 BSS throughputs ──
    folder_413 = THR.get("4.1.3", os.path.join(ROOT, "thr_runs_4_1_3"))
    if os.path.isdir(folder_413):
        for mode_id, mode_label, color in [("EMLSR", "EMLSR", "#3b1f70"),
                                            ("STR",   "STR",   "#3a3f8b")]:
            jfis = []
            pat = re.compile(rf"^{mode_id}_rng\d+_result\.csv$")
            for fn in os.listdir(folder_413):
                if not pat.match(fn):
                    continue
                cols = _load_csv_cols(os.path.join(folder_413, fn))
                if not cols or len(cols) < 3:
                    continue
                bsss = cols[:3]
                if all(v > 0 for v in bsss):
                    s  = sum(bsss)
                    ss = sum(v * v for v in bsss)
                    jfis.append((s * s) / (3 * ss))
            if jfis:
                rows.append((f"Throughput\nScenario 3 ({mode_label})", mean(jfis)))
                colors.append(color)

    # ── MLO and Legacy Stations (4.3): per-run jfi_tot column ──────────────
    folder_43 = os.path.join(ROOT, "thr_runs_4_3")
    if os.path.isdir(folder_43):
        for case_id, case_label, color in [("CaseA", "EMLSR", "#2c6b85"),
                                             ("CaseB", "STR",   "#2A9D8F")]:
            jfis = []
            pat = re.compile(rf"^{case_id}_Pct\d+_rng\d+_result\.csv$")
            for fn in os.listdir(folder_43):
                if not pat.match(fn):
                    continue
                cols = _load_csv_cols(os.path.join(folder_43, fn))
                if not cols or len(cols) < 6:
                    continue
                # 4.3 CSV schema: agg_leg, agg_mld, agg_tot, avg_leg, avg_mld,
                #                 jfi_tot, jfi_leg, jfi_mld
                jfi_tot = cols[5]
                if 0.0 < jfi_tot <= 1.0:
                    jfis.append(jfi_tot)
            if jfis:
                rows.append((f"MLO and Legacy\nStations ({case_label})", mean(jfis)))
                colors.append(color)

    if not rows:
        print("[skip] 4.21 no data", file=sys.stderr)
        plt.close(fig)
        return

    xs = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.bar(xs, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("Scenarios")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "4_21")


def main() -> int:
    print("[load] 4.1.1 baseline ...")
    thr_baseline = load_4_1_1()
    print(f"  -> {len(thr_baseline)} (links,nm) keys")
    print("[load] 4.1.4 OBSS ...")
    thr_obss = load_4_1_4_obss()
    print(f"  -> modes={list(thr_obss.keys())}")

    plot_4_18(thr_baseline, thr_obss)
    plot_4_19()
    plot_4_20()
    plot_4_21()
    return 0


if __name__ == "__main__":
    sys.exit(main())
