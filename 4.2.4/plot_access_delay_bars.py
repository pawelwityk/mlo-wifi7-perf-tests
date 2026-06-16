#!/usr/bin/env python3
"""
plot_access_delay_bars.py -- Scenario 4.2.4 experimental channel-access bars.

Reads per-run CSVs from thr_runs_4_2_4/ and generates:
  plots/4_2_4_access_delay_bars.svg

Bar style mirrors legacy 4.2.4 bars:
- stacked p50 / (p95-p50) / (p99-p95)
- log-scale y-axis
- p99 CI whiskers and value labels
"""

# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim

import csv
import os
import re
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUTPUT_DIR = "thr_runs_4_2_4"
PLOT_DIR = "plots"
PLOT_BARS = os.path.join(PLOT_DIR, "4_2_4_access_delay_bars.svg")

NUM_LINKS = [1, 2, 4]
BAR_LOADS = [100, 1000, 2500]
STRICT_FILTER_POINTS = {(4, 1000), (4, 2500)}
STRICT_TRIM_FRACTION = 0.20
STRICT_WINSOR_FRACTION = 0.30

T95_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

MODE_COLORS = {1: "#2E75D8", 2: "#E63B0E", 4: "#F0B400"}
MODE_LABELS = {1: "SL", 2: "STR:2", 4: "STR:4"}



def _parse_row(text: str):
    """Return (mean, p50, p95, p99, thr) or None."""
    reader = csv.reader([text])
    row = next(reader, [])
    parts = [p.strip() for p in row if p.strip()]
    if len(parts) >= 5:
        return tuple(float(p) for p in parts[:5])
    if len(parts) >= 4:
        avg, p50, p99, thr = (float(p) for p in parts[:4])
        return avg, p50, p99, p99, thr
    if len(parts) >= 2:
        avg, thr = float(parts[0]), float(parts[1])
        return avg, avg, avg, avg, thr
    return None



def load_all_results():
    """Return raw[(nl, load)] = [(rng, avg, p50, p95, p99, thr)]."""
    raw = {}
    if not os.path.isdir(OUTPUT_DIR):
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
            avg, p50, p95, p99, thr = parsed
            raw.setdefault((nl, load), []).append((rng, avg, p50, p95, p99, thr))
        except Exception:
            continue

    return raw



def remove_outliers(values):
    """Two-stage IQR + MAD outlier removal (delegates to _common/_robust)."""
    from _robust import remove_outliers_robust
    return remove_outliers_robust(values)


def filter_rows_for_bars(rows, nl=None, load=None):
    """Filter outlier rows using p99 as the anchor metric.

    Using row-level filtering keeps p50/p95/p99 from the same run together,
    so stacked bar segments remain statistically consistent.
    """
    if len(rows) < 6:
        return rows

    p99_vals = [r[4] for r in rows]

    # Pass 1: IQR filter.
    s = sorted(p99_vals)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    iqr = q3 - q1
    if iqr > 0:
        lo_iqr = q1 - 1.5 * iqr
        hi_iqr = q3 + 1.5 * iqr
        rows_iqr = [r for r in rows if lo_iqr <= r[4] <= hi_iqr]
    else:
        rows_iqr = rows

    # Pass 2: MAD filter to catch heavy-tail points that survive IQR.
    p99_iqr = [r[4] for r in rows_iqr]
    if len(p99_iqr) < 6:
        return rows_iqr if rows_iqr else rows

    med = float(np.median(p99_iqr))
    mad = float(np.median([abs(v - med) for v in p99_iqr]))
    if mad == 0.0:
        return rows_iqr

    # 0.6745 scales MAD to z-score for normal data.
    rows_mad = [r for r in rows_iqr if abs(0.6745 * (r[4] - med) / mad) <= 2.8]

    filtered = rows_mad if len(rows_mad) >= 5 else rows_iqr

    # Targeted strict trim for known heavy-tail points.
    if (nl, load) in STRICT_FILTER_POINTS and len(filtered) >= 8:
        by_p99 = sorted(filtered, key=lambda r: r[4])
        k = int(len(by_p99) * STRICT_TRIM_FRACTION)
        if k > 0 and (len(by_p99) - 2 * k) >= 5:
            filtered = by_p99[k:len(by_p99) - k]

    return filtered



def ci95(values):
    n = len(values)
    if n <= 1:
        return 0.0
    t = T95_CRITICAL.get(n - 1, 1.96)
    return t * stdev(values) / (n ** 0.5)


def winsorize(values, frac):
    """Clamp tails to percentile boundaries (robust against extreme points)."""
    if len(values) < 4 or frac <= 0.0:
        return values
    arr = np.array(values, dtype=float)
    lo = float(np.quantile(arr, frac))
    hi = float(np.quantile(arr, 1.0 - frac))
    return np.clip(arr, lo, hi).tolist()



def plot_bars(raw):
    bar_loads = [l for l in BAR_LOADS if any(raw.get((nl, l)) for nl in NUM_LINKS)]
    load_gbps = [f"{l/1000:.1f}".rstrip("0").rstrip(".") for l in bar_loads]
    if not bar_loads:
        print("[warn] No matching bar loads found in data")
        return

    n_loads = len(bar_loads)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(n_loads)
    width = 0.22
    offsets = {1: -width, 2: 0.0, 4: width}

    for nl in NUM_LINKS:
        p50_vals, p95_delta, p99_delta, p99_total, p99_ci = [], [], [], [], []

        for load in bar_loads:
            rows = raw.get((nl, load), [])
            if rows:
                # For problematic STR:4 points, keep all runs and winsorize tails
                # instead of dropping many samples, which stabilizes CI better.
                if (nl, load) in STRICT_FILTER_POINTS:
                    rows_f = rows
                else:
                    rows_f = filter_rows_for_bars(rows, nl=nl, load=load)

                p50 = mean([r[2] for r in rows_f])
                p95 = mean([r[3] for r in rows_f])
                p99s = [r[4] for r in rows_f]
                if (nl, load) in STRICT_FILTER_POINTS:
                    p99s = winsorize(p99s, STRICT_WINSOR_FRACTION)
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
        c = MODE_COLORS[nl]

        ax.bar(xpos, p50_vals, width=width, color=c, edgecolor="black", linewidth=0.4, label=MODE_LABELS[nl], zorder=3)
        ax.bar(xpos, p95_delta, width=width, bottom=p50_vals, color=c, alpha=0.65, edgecolor="black", linewidth=0.2, zorder=3)
        ax.bar(
            xpos,
            p99_delta,
            width=width,
            bottom=np.array(p50_vals) + np.array(p95_delta),
            color=c,
            alpha=0.35,
            edgecolor="black",
            linewidth=0.2,
            zorder=3,
        )

        for xi, p99, err in zip(xpos, p99_total, p99_ci):
            top = p99 + err
            ax.errorbar(xi, p99, yerr=err, fmt="none", color="black", capsize=4, linewidth=1.1, zorder=5)
            ax.text(xi, top * 1.25, f"{p99:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_title("Scenario 4.2.4 (Experimental) - Channel Access Delay Percentiles")
    ax.set_yscale("log")
    # Top headroom so value labels (placed at (p99+err)*1.25) fit inside the
    # frame. Log scale — multiply current top by a constant.
    _, _ytop = ax.get_ylim()
    ax.set_ylim(bottom=0.001, top=_ytop * 1.6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels(load_gbps)
    ax.set_xlabel("Total traffic load [Gbit/s]")
    ax.set_ylabel("Channel access delay: 50/95/99 percentile [ms]")
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.35, zorder=0)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(PLOT_BARS, bbox_inches="tight", format="svg")
    # Also produce a PDF alongside the SVG for thesis use.
    pdf_path = PLOT_BARS.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"[plot] saved -> {PLOT_BARS}")
    print(f"[plot] saved -> {pdf_path}")



def main():
    raw = load_all_results()
    total = sum(len(v) for v in raw.values())
    print(f"[info] Loaded {total} run results from {OUTPUT_DIR}/")

    if total == 0:
        print("[warn] No data found")
        return

    plot_bars(raw)


if __name__ == "__main__":
    main()
