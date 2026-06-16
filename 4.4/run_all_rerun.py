#!/usr/bin/env python3
"""
Scenario 4.4 — MLO in VR Applications.

Sweep:
  mode      : SLO, MLO
  direction : DL, UL
  mcs       : 0 .. 13
  bw        : 20, 40, 80, 160, 320 MHz
  RngRun    : 1 .. RUNS

Each combination is run RUNS times.  Per-run results stored as one CSV row
in ``thr_runs_4_4/``.

After the sweep, computes the *minimum* (MCS, BW) combination per
(mode, direction) that meets the Wi-Fi Alliance VR-gaming requirements:
  DL: p75 latency < 5 ms
  UL: p90 latency < 2 ms

Plotted as two grids in the style of thesis Fig 4.17a / 4.17b.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from statistics import mean, median

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scratch", "_common"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

import plot_style  # noqa: E402
from _robust import remove_outliers_robust  # noqa: E402

plot_style.install()

# ---------------------------------------------------------------------------
NS3_BINARY = os.path.join(ROOT, "build", "scratch", "4.4", "ns3.45-wifi-vr-mlo-optimized")
OUTPUT_DIR = os.path.join(ROOT, "thr_runs_4_4")

MODES       = ["SLO", "MLO"]
DIRECTIONS  = ["DL", "UL"]
MCS_LIST    = list(range(0, 14))           # 0..13
BW_LIST     = [20, 40, 80, 160, 320]       # MHz
RUNS        = int(os.environ.get("RUNS_4_4", "5"))
SIM_TIME    = float(os.environ.get("SIMTIME_4_4", "6.0"))

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "6"))

REQ_DL_P75_MS = 5.0   # Wi-Fi Alliance VR-gaming requirement
REQ_UL_P90_MS = 2.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

REGEX = {
    "p50":  re.compile(r"VR p50 lat:\s*([0-9.eE+-]+)\s*ms"),
    "p75":  re.compile(r"VR p75 lat:\s*([0-9.eE+-]+)\s*ms"),
    "p90":  re.compile(r"VR p90 lat:\s*([0-9.eE+-]+)\s*ms"),
    "p99":  re.compile(r"VR p99 lat:\s*([0-9.eE+-]+)\s*ms"),
    "mean": re.compile(r"VR mean lat:\s*([0-9.eE+-]+)\s*ms"),
    "rx":   re.compile(r"VR rx pkts:\s*(\d+)"),
}
FIELDS = list(REGEX.keys())


def _path(mode: str, direction: str, mcs: int, bw: int, rng: int) -> str:
    return os.path.join(OUTPUT_DIR,
                        f"{mode}_{direction}_mcs{mcs:02d}_bw{bw:03d}_rng{rng:03d}_result.csv")


def _parse(text: str) -> dict | None:
    out = {}
    for k, rx in REGEX.items():
        m = rx.search(text)
        if not m:
            return None
        out[k] = float(m.group(1)) if k != "rx" else int(m.group(1))
    return out


def run_job(mode: str, direction: str, mcs: int, bw: int, rng: int):
    fp = _path(mode, direction, mcs, bw, rng)
    if os.path.exists(fp):
        try:
            with open(fp) as f:
                row = next(csv.reader(f))
            if len(row) >= len(FIELDS):
                return mode, direction, mcs, bw, rng, dict(zip(FIELDS, [float(x) for x in row]))
        except Exception:
            pass

    cmd = [NS3_BINARY,
           f"--mode={mode}",
           f"--direction={direction}",
           f"--mcs={mcs}",
           f"--channelWidth={bw}",
           f"--simTime={SIM_TIME}",
           f"--RngRun={rng}"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        parsed = _parse(proc.stdout)
        if parsed is None:
            print(f"  PARSE-FAIL {mode} {direction} mcs{mcs} bw{bw} rng{rng}",
                  file=sys.stderr)
            print(proc.stdout[-300:], file=sys.stderr)
            return mode, direction, mcs, bw, rng, None
        with open(fp, "w") as f:
            f.write(",".join(str(parsed[k]) for k in FIELDS) + "\n")
        return mode, direction, mcs, bw, rng, parsed
    except Exception as e:
        print(f"  ERROR {mode} {direction} mcs{mcs} bw{bw} rng{rng}: {e}",
              file=sys.stderr)
        return mode, direction, mcs, bw, rng, None


def sweep():
    if not os.path.exists(NS3_BINARY):
        print(f"[ERROR] simulator not built: {NS3_BINARY}", file=sys.stderr)
        print("        ./ns3 build wifi-vr-mlo", file=sys.stderr)
        return

    jobs = list(product(MODES, DIRECTIONS, MCS_LIST, BW_LIST, range(1, RUNS + 1)))
    print(f"[run] {len(jobs)} jobs ({MAX_WORKERS} workers)")
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(run_job, *j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            if done % 25 == 0:
                print(f"  ... {done}/{len(jobs)}", flush=True)


def load_all() -> dict:
    """{(mode, dir, mcs, bw) -> {field -> [values]}}."""
    out: dict = {}
    pat = re.compile(r"^(SLO|MLO)_(DL|UL)_mcs(\d{2})_bw(\d{3})_rng\d+_result\.csv$")
    if not os.path.isdir(OUTPUT_DIR):
        return out
    for fn in os.listdir(OUTPUT_DIR):
        m = pat.match(fn)
        if not m:
            continue
        key = (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))
        try:
            with open(os.path.join(OUTPUT_DIR, fn)) as f:
                row = next(csv.reader(f))
            if len(row) < len(FIELDS):
                continue
            d = {k: float(v) for k, v in zip(FIELDS, row)}
        except Exception:
            continue
        bucket = out.setdefault(key, {f: [] for f in FIELDS})
        for k in FIELDS:
            bucket[k].append(d[k])
    return out


def _meets(samples: list, threshold: float) -> bool:
    if not samples:
        return False
    samples = remove_outliers_robust(samples)
    if not samples:
        return False
    return median(samples) <= threshold


def _save(fig, name):
    p = os.path.join(plot_style.PLOT_DIR, name)
    fig.savefig(p + ".svg", format="svg")
    fig.savefig(p + ".pdf", format="pdf")
    print(f"[plot] {p}.{{svg,pdf}}")
    plt.close(fig)


def plot_grid(data: dict, direction: str, percentile_field: str,
              threshold: float, title: str, fname: str):
    """Render a praca-style filled-region plot showing the (MCS, BW) combos
    that meet the latency requirement, separately for SLO and MLO.

    Mirrors thesis Fig 4.17a (DL, 5 ms p75) / 4.17b (UL, 2 ms p90):
      - X axis: MCS (decreasing 13 → 0)
      - Y axis: channel bandwidth (log scale, 20…320 MHz)
      - For each mode, a step boundary curve marks the lowest MCS that still
        meets the requirement at each bandwidth; the region of *meeting*
        combos (boundary → MCS_max) is filled with a translucent colour.
    """
    nbw = len(BW_LIST)
    nmcs = len(MCS_LIST)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    mode_styles = {
        "SLO": {"line": "#7B5BA1", "fill": "#7B5BA1", "alpha": 0.18},
        "MLO": {"line": "#2A9D8F", "fill": "#2A9D8F", "alpha": 0.18},
    }

    mcs_max = max(MCS_LIST)

    for mode in MODES:
        # For each BW, find the *lowest* MCS that still meets the requirement.
        # If no MCS meets at this BW, skip (no fill at that BW).
        boundary = []  # list of (bw, mcs_min) pairs in MCS order
        for bw in BW_LIST:
            mcs_min = None
            for mcs in sorted(MCS_LIST):
                samples = data.get((mode, direction, mcs, bw), {}).get(percentile_field, [])
                if _meets(samples, threshold):
                    mcs_min = mcs
                    break
            if mcs_min is not None:
                boundary.append((bw, mcs_min))

        if not boundary:
            continue

        bws  = np.array([b for b, _ in boundary], dtype=float)
        mins = np.array([m for _, m in boundary], dtype=float)

        st = mode_styles[mode]

        # Smooth boundary in MCS-vs-log(BW) space — linear-interpolate the
        # discrete (BW, MCS_min) points so the line and filled region show
        # continuous diagonals instead of harsh per-BW step edges.
        log_bws = np.log10(bws)
        log_dense = np.linspace(log_bws.min(), log_bws.max(), 200)
        mins_dense = np.interp(log_dense, log_bws, mins)
        bws_dense  = np.power(10.0, log_dense)
        ax.fill_betweenx(bws_dense, mins_dense, mcs_max,
                         color=st["fill"], alpha=st["alpha"],
                         linewidth=0, zorder=2)
        # Compose boundary path: horizontal closure at the lowest meeting BW
        # (from MCS_max down to the first boundary MCS), then the smooth
        # diagonal up through the dense interpolation.
        line_x = np.concatenate(([mcs_max], mins_dense))
        line_y = np.concatenate(([bws_dense[0]], bws_dense))
        ax.plot(line_x, line_y, color=st["line"], linewidth=1.8,
                label=mode, zorder=3)

    ax.set_xlim(mcs_max + 0.5, -0.5)             # MCS descending
    ax.set_xticks(MCS_LIST)
    ax.set_xticklabels([str(m) for m in MCS_LIST])
    ax.set_xlabel("MCS")

    ax.set_yscale("log")
    ax.set_yticks(BW_LIST)
    ax.set_yticklabels([f"{bw} MHz" for bw in BW_LIST])
    ax.set_ylim(min(BW_LIST) * 0.85, max(BW_LIST) * 1.15)
    ax.set_ylabel("Channel bandwidth")

    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    _save(fig, fname)


def main():
    args = sys.argv[1:]
    if "--plot-only" not in args:
        sweep()
    data = load_all()
    n = sum(len(v["p50"]) for v in data.values())
    print(f"[load] {len(data)} configs, {n} samples")
    if n == 0:
        print("[warn] no data — nothing to plot", file=sys.stderr)
        return
    plot_grid(data, "DL", "p75", REQ_DL_P75_MS,
              "Fig 4.17a — VR DL: combinations meeting 5 ms p75 requirement",
              "4_17_a")
    plot_grid(data, "UL", "p90", REQ_UL_P90_MS,
              "Fig 4.17b — VR UL: combinations meeting 2 ms p90 requirement",
              "4_17_b")


if __name__ == "__main__":
    main()
