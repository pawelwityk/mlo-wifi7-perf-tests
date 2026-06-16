#!/usr/bin/env python3
"""
plot_delay_vs_load.py — Latency Scenario #1 (4.2.1)

Reads pre-collected CSVs from thr_runs_4_2_1/ and produces
delay-vs-normalized-traffic-load plots.

X axis : normalized traffic load ρ from scenario configuration
Y axis : mean DL latency [ms]
Lines  : 6 scenarios  —  (numLinks, nStas) ∈ {1,2} × {1,4,10}
         each line has two data points corresponding to channelWidth 80 and 160 MHz

EHT MCS 11, 2 SS, GI 800 ns PHY peak data rates (per link):
    80 MHz  → 1201.1 Mb/s
   160 MHz  → 2402.2 Mb/s
"""

# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR   = "thr_runs_4_2_1"
PLOT_DIR     = "plots"
PLOT_FILES   = {
    80: os.path.join(PLOT_DIR, "4_2_1_80.svg"),
    160: os.path.join(PLOT_DIR, "4_2_1_160.svg"),
}

NUM_LINKS    = [1, 2]
CHAN_WIDTHS   = [80, 160]
NSTA_VALUES  = [1, 4, 10]
RUNS         = 10
RNG_START    = 1001

CI           = 0.95
RHO_LEVELS   = [round(0.1 * i, 1) for i in range(0, 11)]
PLOT_RHO_LEVELS = [round(0.1 * i, 1) for i in range(1, 11)]
ALLOW_LEGACY_FILES = False
LATENCY_YMAX_MS = None  # Auto-scale (symlog handles range)
LATENCY_LINTHRESH = 0.5  # Linear below this, log above
LATENCY_YMAX_PERCENTILE = 95

os.makedirs(PLOT_DIR, exist_ok=True)

# EHT MCS 11, 2 SS, GI 800 ns — peak PHY data-rate per link [Mbit/s]
PHY_PEAK = {80: 1201.1, 160: 2402.2}

# ── Helpers ────────────────────────────────────────────────────────────────

def _result_path(nl, cw, ns, rho_tag, rng):
    return os.path.join(OUTPUT_DIR, f"L{nl}_W{cw}_S{ns}_{rho_tag}_rng{rng}_result.csv")


def load_all_results():
    """Return (raw, sat, new_count, old_count). sat[nl][cw][ns][rho] is True
    when at least one .saturated sentinel exists for that config."""
    raw = {
        nl: {cw: {ns: {rho: {"lat": [], "thr": []}
                       for rho in RHO_LEVELS}
                  for ns in NSTA_VALUES}
             for cw in CHAN_WIDTHS}
        for nl in NUM_LINKS
    }
    sat = {
        nl: {cw: {ns: {rho: False for rho in RHO_LEVELS}
                  for ns in NSTA_VALUES}
             for cw in CHAN_WIDTHS}
        for nl in NUM_LINKS
    }
    pattern_new = re.compile(r"^L(\d+)_W(\d+)_S(\d+)_R(\d+)_rng(\d+)_result\.csv$")
    pattern_old = re.compile(r"^L(\d+)_W(\d+)_S(\d+)_rng(\d+)_result\.csv$")
    pattern_sat = re.compile(r"^L(\d+)_W(\d+)_S(\d+)_R(\d+)_rng(\d+)_saturated$")
    new_count = 0
    old_count = 0
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        m_sat = pattern_sat.match(fname)
        if m_sat:
            nl = int(m_sat.group(1)); cw = int(m_sat.group(2))
            ns = int(m_sat.group(3)); rho = int(m_sat.group(4)) / 10.0
            if (nl in NUM_LINKS and cw in CHAN_WIDTHS
                    and ns in NSTA_VALUES and rho in sat[nl][cw][ns]):
                sat[nl][cw][ns][rho] = True
            continue

        m_new = pattern_new.match(fname)
        m_old = pattern_old.match(fname)
        if not m_new and not m_old:
            continue

        if m_new:
            new_count += 1
            nl = int(m_new.group(1))
            cw = int(m_new.group(2))
            ns = int(m_new.group(3))
            rho = int(m_new.group(4)) / 10.0
        else:
            old_count += 1
            if not ALLOW_LEGACY_FILES:
                continue
            nl = int(m_old.group(1))
            cw = int(m_old.group(2))
            ns = int(m_old.group(3))
            rho = None

        if nl not in NUM_LINKS or cw not in CHAN_WIDTHS or ns not in NSTA_VALUES:
            continue

        try:
            with open(os.path.join(OUTPUT_DIR, fname)) as f:
                lat, thr = map(float, f.read().strip().split(","))

            if rho is None:
                capacity = nl * PHY_PEAK[cw]
                rho = thr / capacity

            rho = min(RHO_LEVELS, key=lambda x: abs(x - rho))
            raw[nl][cw][ns][rho]["lat"].append(lat)
            raw[nl][cw][ns][rho]["thr"].append(thr)
        except Exception as exc:
            print(f"[warn] Could not read {fname}: {exc}")
    return raw, sat, new_count, old_count


def ci_mean(data):
    """Return (mean, half-width of 95 % CI) after IQR+MAD outlier removal."""
    from _robust import ci_hw_robust
    return ci_hw_robust(data, confidence=CI)


def _compute_ymax(lat_values, lat_err_values):
    """Return a y-axis maximum that contains every plotted data point.

    The previous implementation clipped to a percentile and lost data
    above the 95th percentile; we now use the global max + headroom.
    """
    if LATENCY_YMAX_MS is not None:
        return float(LATENCY_YMAX_MS)
    if not lat_values:
        return 1.0
    upper = [v + e for v, e in zip(lat_values, lat_err_values)]
    return max(1.0, max(upper) * 1.10)


# ── Plot ───────────────────────────────────────────────────────────────────

def plot(raw, _sat=None):
    # One colour + linestyle per numLinks; one marker per nStas
    link_styles = {
        1: dict(color="#4C72B0", linestyle="-"),
        2: dict(color="#DD8452", linestyle="--"),
    }
    sta_markers = {1: "o", 4: "s", 10: "^"}

    print(f"\n{'Config':<20}  {'norm_load':>10}  {'lat_mean [ms]':>14}  {'n':>4}")
    print("-" * 56)

    for cw in sorted(CHAN_WIDTHS):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        handles = []
        labels = []
        all_ys = []
        all_yerrs = []

        for nl in NUM_LINKS:
            style = link_styles[nl]
            for ns in NSTA_VALUES:
                xs, ys, yerrs = [], [], []
                for rho in PLOT_RHO_LEVELS:
                    data = raw[nl][cw][ns][rho]
                    if not data["lat"]:
                        continue
                    lat_mean, lat_h = ci_mean(data["lat"])
                    xs.append(rho); ys.append(lat_mean); yerrs.append(lat_h)
                    all_ys.append(lat_mean); all_yerrs.append(lat_h)
                    print(f"L{nl} W{cw:3d} S{ns:2d}           "
                          f"{rho:10.4f}  {lat_mean:14.3f}  "
                          f"{len(data['lat']):4d}")

                if not xs:
                    continue

                scenario_label = f"{nl} link, {ns} user"
                h = ax.errorbar(
                    xs, ys, yerr=yerrs,
                    color=style["color"],
                    marker=sta_markers[ns],
                    linestyle="-",
                    markersize=6,
                    linewidth=1.5,
                    capsize=5, capthick=1.6, elinewidth=1.6,
                    ecolor=style["color"],
                    zorder=3,
                )
                handles.append(h)
                labels.append(scenario_label)

        ax.set_xlabel("Normalized traffic load",
                      fontsize=12)
        ax.set_title(
            f"Channel width: {cw} MHz\n"
            "5 GHz, EHT MCS 11, A-MPDU 1024  |  "
            f"Mean ± {int(CI*100)}% CI",
            fontsize=11,
        )
        ax.set_xlim(0.07, 1.03)
        ax.set_xticks(PLOT_RHO_LEVELS)
        # Y-axis sized so the full data range (up to ~6.7 ms at saturation)
        # fits without clipping.
        ax.set_ylim(0.0, 8.0)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
        ax.xaxis.grid(True, linestyle="--", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel("Channel access delay [ms]", fontsize=12)
        ax.legend(handles, labels, title="Scenario", fontsize=9, title_fontsize=9,
                  loc='upper left', framealpha=0.9)

        fig.suptitle(
            "Latency Scenario #1 — Delay vs. Normalized Traffic Load",
            fontsize=13,
            y=1.01,
        )

        plt.tight_layout()
        out = PLOT_FILES[cw]
        plt.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"\n[OK] Plot saved to {out}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_DIR):
        raise SystemExit(f"[error] Results directory not found: {OUTPUT_DIR!r}")

    raw, sat, new_count, old_count = load_all_results()
    if new_count == 0 and old_count > 0 and not ALLOW_LEGACY_FILES:
        raise SystemExit(
            "[error] Found only legacy 4.2.1 results (no rho-tagged files). "
            "Run 'python3 scratch/4.2.1/run_all_rerun.py' to regenerate data "
            "with normalized-load bins before plotting."
        )
    has_data = any(
        raw[nl][cw][ns][rho]["lat"]
        for nl in NUM_LINKS
        for cw in CHAN_WIDTHS
        for ns in NSTA_VALUES
        for rho in RHO_LEVELS
    )
    if not has_data:
        raise SystemExit(
            "[error] No 4.2.1 result samples found. Run 'python3 scratch/4.2.1/run_all_rerun.py' first."
        )
    plot(raw, sat)
