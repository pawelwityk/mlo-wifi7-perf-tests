#!/usr/bin/env python3
"""
run_all_rerun.py  —  Latency Scenario #3 (4.2.3)
Sweep: numLinks × MLO mode (STR/EMLSR) × offeredLoad × RngRun
Persistence: per-run CSV result files (resume after crash / add more runs).
Re-run: failed jobs retried sequentially after the parallel pass.
"""

# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim

import os
import re
import subprocess
import sys
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from statistics import mean, stdev

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
NS3_BINARY   = "build/scratch/4.2.3/ns3.45-wifi-mlo-latency-scenario3-optimized"
OUTPUT_DIR   = "thr_runs_4_2_3"
PLOT_DIR     = "plots"
PLOT_FILE    = os.path.join(PLOT_DIR, "4_2_3.svg")

NUM_LINKS      = [1, 2, 3]
MLO_MODES      = ["STR", "EMLSR"]         # EMLSR is valid for numLinks >= 2
OFFERED_LOADS  = list(range(100, 2600, 100))   # 100 … 2500 Mb/s  (25 points)
RUNS           = 20
SIM_TIME       = 20.0
PAYLOAD        = 1500
NMDPUS         = 1024

MAX_WORKERS    = 11
RERUN_MISSING  = True
NORMALIZED_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
PLOT_RNG_RUNS = list(range(1, RUNS + 1))

# Two-sided 95% Student-t critical values for df=1..30.
T95_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RE_LAT = re.compile(r"Mean DL Latency:\s*([0-9.]+)\s*ms")
RE_P1 = re.compile(r"DL Latency p1:\s*([0-9.]+)\s*ms")
RE_P99 = re.compile(r"DL Latency p99:\s*([0-9.]+)\s*ms")
RE_THR = re.compile(r"Mean DL Throughput:\s*([0-9.]+)\s*Mb/s")

# ---------------------------------------------------------------------------
#  Persistence helpers
# ---------------------------------------------------------------------------

def _result_path(nl: int, mode: str, load: int, rng: int) -> str:
    fname = f"L{nl}_M{mode}_Load{load}_rng{rng}_result.csv"
    return os.path.join(OUTPUT_DIR, fname)

def _key(nl: int, mode: str, load: int) -> tuple:
    """Result cache key."""
    return (nl, mode, load)

def _mode_to_bool(mode: str) -> bool:
    """Convert mode string to emlsr boolean."""
    return mode == "EMLSR"


def _is_valid_combo(nl: int, mode: str) -> bool:
    """EMLSR requires multi-link operation."""
    return not (mode == "EMLSR" and nl < 2)


def _parse_result_row(text: str, require_p99: bool = False):
    """Parse a result CSV row from either the legacy or current format.

    Legacy files contain ``latency,throughput``.
    Current files contain ``latency,p1,p99,throughput``.
    """
    reader = csv.reader([text])
    row = next(reader, [])
    parts = [item.strip() for item in row if item.strip()]
    if len(parts) >= 4:
        lat, p1, p99, thr = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        return lat, p1, p99, thr
    if len(parts) >= 3:
        lat, p99, thr = float(parts[0]), float(parts[1]), float(parts[2])
        return lat, lat, p99, thr
    if len(parts) >= 2:
        if require_p99:
            return None
        lat, thr = float(parts[0]), float(parts[1])
        return lat, lat, lat, thr
    return None


PLOT_LOADS = [300, 800, 1300, 1800, 2300]

# ---------------------------------------------------------------------------
#  Run a single job + parse output
# ---------------------------------------------------------------------------

def run_job(nl: int, mode: str, load: int, rng: int, force_refresh: bool = False) -> tuple:
    """
    Run: ./ns3 run "scratch/4.2.3/wifi-mlo-latency-scenario3 --numLinks=nl --emlsr=bool \
                                  --offeredLoad=load --RngRun=rng"
    Parse output for latency and throughput.
    Returns: (nl, mode, load, rng, lat_ms, p1_ms, p99_ms, thr_mbps) or
    (nl, mode, load, rng, None, None, None, None) on failure.
    """
    result_file = _result_path(nl, mode, load, rng)
    if os.path.exists(result_file) and not force_refresh:
        try:
            with open(result_file) as f:
                parsed = _parse_result_row(f.read().strip())
            if parsed is not None:
                lat, p1, p99, thr = parsed
                return (nl, mode, load, rng, lat, p1, p99, thr)
        except Exception:
            pass

    emlsr_bool = _mode_to_bool(mode)
    cmd = [
        NS3_BINARY,
        f"--simTime={SIM_TIME}",
        f"--payloadSize={PAYLOAD}",
        f"--nMpdus={NMDPUS}",
        f"--numLinks={nl}",
        f"--emlsr={str(emlsr_bool).lower()}",
        f"--offeredLoad={load}",
        f"--RngRun={rng}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        lat_m = RE_LAT.search(result.stdout)
        p1_m = RE_P1.search(result.stdout)
        p99_m = RE_P99.search(result.stdout)
        thr_m = RE_THR.search(result.stdout)
        if lat_m and p1_m and p99_m and thr_m:
            lat = float(lat_m.group(1))
            p1 = float(p1_m.group(1))
            p99 = float(p99_m.group(1))
            thr = float(thr_m.group(1))
            # Persist result
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(result_file, 'w') as f:
                f.write(f"{lat},{p1},{p99},{thr}\n")
            return (nl, mode, load, rng, lat, p1, p99, thr)
    except Exception as e:
        print(f"  ERROR L{nl} M{mode} Load{load} rng{rng}: {e}", file=sys.stderr)

    return (nl, mode, load, rng, None, None, None, None)

# ---------------------------------------------------------------------------
#  Load existing results
# ---------------------------------------------------------------------------

def load_existing_results(require_p99: bool = False):
    """Load existing result files into a dict keyed by (nl, mode, load).

    If *require_p99* is true, skip legacy two-field CSVs and keep only files
    that already contain the packet-level percentile columns.
    """
    results = {}
    if not os.path.isdir(OUTPUT_DIR):
        return results

    for fname in os.listdir(OUTPUT_DIR):
        m = re.match(r"L(\d+)_M(\w+)_Load(\d+)_rng(\d+)_result\.csv$", fname)
        if not m:
            continue
        nl, mode, load, rng = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        key = _key(nl, mode, load)
        fpath = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(fpath) as f:
                parsed = _parse_result_row(f.read().strip(), require_p99=require_p99)
                if parsed is None:
                    continue
                lat, p1, p99, thr = parsed
            if key not in results:
                results[key] = []
            results[key].append((rng, lat, p1, p99, thr))
        except Exception:
            pass

    return results


def refresh_plot_data():
    """Refresh only the loads used by the plot, selected per configuration."""
    print("[*] Refreshing plot loads with packet-level percentile data...", flush=True)
    base_results = load_existing_results(require_p99=False)
    selected_loads = _select_plot_loads(base_results)
    jobs = []
    for nl in NUM_LINKS:
        for mode in MLO_MODES:
            if not _is_valid_combo(nl, mode):
                continue
            for load in selected_loads.get((nl, mode), []):
                for rng in PLOT_RNG_RUNS:
                    jobs.append((nl, mode, load, rng))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_job, nl, mode, load, rng, True) for nl, mode, load, rng in jobs]
        for future in as_completed(futures):
            future.result()


def _select_plot_loads(raw):
    """Pick representative offered-load points for each configuration.

    The selection is based on each configuration's own throughput scale so the
    resulting plot compares comparable operating regions.
    """
    selected = {}
    for nl, mode in [(1, "STR"), (2, "EMLSR"), (2, "STR"), (3, "EMLSR"), (3, "STR")]:
        if not _is_valid_combo(nl, mode):
            continue

        load_to_thr = {}
        for (key_nl, key_mode, load), rows in raw.items():
            if key_nl != nl or key_mode != mode or not rows:
                continue
            load_to_thr[load] = mean(row[4] for row in rows)

        if not load_to_thr:
            continue

        capacity = max(load_to_thr.values())
        chosen = []
        used = set()
        for fraction in NORMALIZED_FRACTIONS:
            target = capacity * fraction
            best_load = min(
                load_to_thr,
                key=lambda load: (abs(load_to_thr[load] - target), load),
            )
            if best_load in used:
                remaining = [load for load in load_to_thr if load not in used]
                if remaining:
                    best_load = min(
                        remaining,
                        key=lambda load: (abs(load_to_thr[load] - target), load),
                    )
            used.add(best_load)
            chosen.append(best_load)

        selected[(nl, mode)] = chosen

    return selected


def plot_results(raw):
    """Create a grouped-bar SVG that resembles the reference latency figure."""
    def escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def color_for(nl: int, mode: str) -> str:
        palette = {
            (1, "STR"): "#6d1b7b",      # purple
            (2, "EMLSR"): "#5a73a8",    # blue
            (2, "STR"): "#3b9ea1",      # teal
            (3, "EMLSR"): "#6fd36f",    # green
            (3, "STR"): "#f7df2e",      # yellow
        }
        return palette[(nl, mode)]

    def ci95(values):
        from _robust import ci_hw_robust
        return ci_hw_robust(values, confidence=0.95)[1]

    series_order = [(1, "STR"), (2, "EMLSR"), (2, "STR"), (3, "EMLSR"), (3, "STR")]
    series_names = {
        (1, "STR"): "SL 1 link 99%-tile",
        (2, "EMLSR"): "EMLSR 2 links 99%-tile",
        (2, "STR"): "STR 2 links 99%-tile",
        (3, "EMLSR"): "EMLSR 3 links 99%-tile",
        (3, "STR"): "STR 3 links 99%-tile",
    }

    series_data = {key: [] for key in series_order}
    # Per-bar overlays: (series, fraction) -> (mean_value, p1_value)
    per_bar_overlay = {}
    selected_loads = _select_plot_loads(raw)

    x_labels = [f"{int(fraction * 100)}%" for fraction in NORMALIZED_FRACTIONS]

    for idx, fraction in enumerate(NORMALIZED_FRACTIONS):
        for nl, mode in series_order:
            chosen_loads = selected_loads.get((nl, mode), [])
            if idx >= len(chosen_loads):
                continue
            load = chosen_loads[idx]
            key = (nl, mode, load)
            if key not in raw or len(raw[key]) == 0:
                continue
            mean_samples = [row[1] for row in raw[key]]
            p1_samples = [row[2] for row in raw[key]]
            p99_samples = [row[3] for row in raw[key]]
            series_data[(nl, mode)].append((fraction, mean(p99_samples), ci95(p99_samples)))
            per_bar_overlay[((nl, mode), fraction)] = (
                mean(mean_samples), mean(p1_samples)
            )

    width = 900
    height = 700
    left = 110
    right = 30
    top = 40
    bottom = 85
    chart_w = width - left - right
    chart_h = height - top - bottom

    all_vals = [v for points in series_data.values() for _, v, _ in points]
    if not all_vals:
        all_vals = [0.0]
    max_y = max(all_vals) * 1.18
    if per_bar_overlay:
        max_y = max(max_y, max(v for v, _ in per_bar_overlay.values()) * 1.18)
        max_y = max(max_y, max(v for _, v in per_bar_overlay.values()) * 1.18)
    max_y = max(max_y, 0.1)

    def x_for(idx: int) -> float:
        return left + chart_w * (idx + 0.5) / len(NORMALIZED_FRACTIONS)

    def y_for(value: float) -> float:
        return top + chart_h - (value / max_y) * (chart_h - 70)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>',
        'text { font-family: Arial, Helvetica, sans-serif; fill: #333; }',
        '.title { font-size: 18px; font-weight: 500; }',
        '.axis-title { font-size: 20px; }',
        '.tick { font-size: 14px; }',
        '.legend { font-size: 15px; }',
        '.grid { stroke: #b8b8b8; stroke-width: 1.0; stroke-dasharray: 5 3; }',
        '.axis { stroke: #d2d2d2; stroke-width: 1.2; }',
        '.frame { stroke: black; stroke-width: 1.2; }',
        '</style>',
    ]

    # grid and y ticks
    n_y_ticks = 8
    y_ticks = [i * max_y / (n_y_ticks - 1) for i in range(n_y_ticks)]
    y_tick_fmt = "{:.0f}" if max_y >= 10 else "{:.1f}"
    for tick in y_ticks:
        y = y_for(tick)
        if y < top or y > top + chart_h:
            continue
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        svg.append(f'<text class="tick" x="{left - 10}" y="{y + 5:.1f}" text-anchor="end">{y_tick_fmt.format(tick)}</text>')

    # vertical separators between groups — dashed, matching horizontal grid
    for idx in range(len(NORMALIZED_FRACTIONS)):
        x = x_for(idx)
        svg.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_h}"/>')
        svg.append(f'<text class="tick" x="{x:.1f}" y="{top + chart_h + 25}" text-anchor="middle">{x_labels[idx]}</text>')

    # Axis frame: close the plot on all 4 edges (top + right were missing).
    svg.append(f'<line class="frame" x1="{left}" y1="{top}" x2="{width - right}" y2="{top}"/>')
    svg.append(f'<line class="frame" x1="{width - right}" y1="{top}" x2="{width - right}" y2="{top + chart_h}"/>')
    svg.append(f'<line class="frame" x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}"/>')
    svg.append(f'<line class="frame" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}"/>')

    svg.append(f'<text class="axis-title" transform="translate(28,{top + chart_h/2:.1f}) rotate(-90)" text-anchor="middle">Channel access delay [ms]</text>')
    svg.append(f'<text class="axis-title" x="{left + chart_w/2:.1f}" y="{height - 20}" text-anchor="middle">Normalized Traffic Load</text>')

    # grouped bars
    bar_width = 16
    offsets = [-32, -16, 0, 16, 32]
    bar_clips = []

    for series_index, key in enumerate(series_order):
        points = series_data[key]
        for idx, fraction in enumerate(NORMALIZED_FRACTIONS):
            matching = next((item for item in points if item[0] == fraction), None)
            if not matching:
                continue
            _, value, err = matching
            x_center = x_for(idx)
            x = x_center + offsets[series_index]
            y = y_for(value)
            base = y_for(0)
            color = color_for(*key)
            bar_clips.append(
                f'<rect x="{x - bar_width/2:.1f}" y="{y:.1f}" width="{bar_width}" height="{base - y:.1f}"/>'
            )
            svg.append(
                f'<rect x="{x - bar_width/2:.1f}" y="{y:.1f}" width="{bar_width}" height="{base - y:.1f}" fill="{color}" stroke="#111111" stroke-width="1"/>'
            )
            y_low = y_for(max(value - err, 0.0))
            y_high = y_for(value + err)
            svg.append(
                f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="#111111" stroke-width="0.7"/>'
            )
            svg.append(f'<line x1="{x - 2.5:.1f}" y1="{y_low:.1f}" x2="{x + 2.5:.1f}" y2="{y_low:.1f}" stroke="#111111" stroke-width="0.7"/>')
            svg.append(f'<line x1="{x - 2.5:.1f}" y1="{y_high:.1f}" x2="{x + 2.5:.1f}" y2="{y_high:.1f}" stroke="#111111" stroke-width="0.7"/>')

    # Per-bar horizontal overlays for Average (thick) and 1%-tile (thin),
    # spanning each bar's width — matches praca_mgr Fig. 4.9 / 4.12 style.
    for series_index, key in enumerate(series_order):
        for idx, fraction in enumerate(NORMALIZED_FRACTIONS):
            overlay = per_bar_overlay.get((key, fraction))
            if overlay is None:
                continue
            mean_v, p1_v = overlay
            x_center = x_for(idx)
            x = x_center + offsets[series_index]
            x_left  = x - bar_width / 2
            x_right = x + bar_width / 2
            y_avg = y_for(mean_v)
            y_p1  = y_for(p1_v)
            svg.append(
                f'<line x1="{x_left:.1f}" y1="{y_avg:.1f}" '
                f'x2="{x_right:.1f}" y2="{y_avg:.1f}" '
                f'stroke="#111111" stroke-width="2.0" stroke-linecap="butt"/>'
            )
            svg.append(
                f'<line x1="{x_left:.1f}" y1="{y_p1:.1f}" '
                f'x2="{x_right:.1f}" y2="{y_p1:.1f}" '
                f'stroke="#111111" stroke-width="1.0" stroke-linecap="butt"/>'
            )

    # legend (10 px gap from the frame top, mirroring the left-side gap)
    legend_x = 128
    legend_y = 82
    legend_entries = [
        ("SL 1 link 99%-tile", color_for(1, "STR"), "bar"),
        ("EMLSR 2 links 99%-tile", color_for(2, "EMLSR"), "bar"),
        ("STR 2 links 99%-tile", color_for(2, "STR"), "bar"),
        ("EMLSR 3 links 99%-tile", color_for(3, "EMLSR"), "bar"),
        ("STR 3 links 99%-tile", color_for(3, "STR"), "bar"),
        ("Average", "#111111", "line-thick"),
        ("1%-tile", "#111111", "line-thin"),
    ]
    svg.append('<rect x="120" y="50" width="345" height="260" fill="white" opacity="0.95" stroke="#808080" stroke-width="1.5"/>')
    for idx, (label, color, kind) in enumerate(legend_entries):
        y = legend_y + idx * 30
        if kind == "line-thick":
            svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 36}" y2="{y}" stroke="{color}" stroke-width="1.9"/>')
        elif kind == "line-thin":
            svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 36}" y2="{y}" stroke="{color}" stroke-width="1.2"/>')
        else:
            svg.append(f'<rect x="{legend_x}" y="{y - 11}" width="36" height="20" fill="{color}" stroke="#111111" stroke-width="1"/>')
        svg.append(f'<text class="legend" x="{legend_x + 46}" y="{y + 5}" text-anchor="start">{escape(label)}</text>')

    svg.append("</svg>")
    with open(PLOT_FILE, "w") as f:
        f.write("\n".join(svg))
    print(f"\n[plot] saved → {PLOT_FILE}", flush=True)
    # Also write PDF sibling so the thesis pipeline picks it up.
    pdf_path = PLOT_FILE[:-4] + ".pdf"
    try:
        import cairosvg
        cairosvg.svg2pdf(url=PLOT_FILE, write_to=pdf_path)
        print(f"[plot] saved → {pdf_path}", flush=True)
    except Exception as exc:
        print(f"[warn] svg→pdf conversion failed: {exc}", flush=True)

# ---------------------------------------------------------------------------
#  Main sweep
# ---------------------------------------------------------------------------

def main():
    print("[*] Loading existing results...", flush=True)
    existing = load_existing_results(require_p99=True)

    print(f"[*] Scheduling jobs: numLinks={NUM_LINKS}, modes={MLO_MODES}, "
          f"loads={len(OFFERED_LOADS)} points, {RUNS} runs each", flush=True)

    # Generate all jobs
    jobs = []
    for nl, mode, load, rng in product(NUM_LINKS, MLO_MODES, OFFERED_LOADS, range(1, RUNS + 1)):
        if _is_valid_combo(nl, mode):
            jobs.append((nl, mode, load, rng))

    print(f"[*] Total jobs: {len(jobs)}", flush=True)

    # Run in parallel
    completed = 0
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_job, nl, mode, load, rng): (nl, mode, load, rng)
                   for nl, mode, load, rng in jobs}
        for future in as_completed(futures):
            completed += 1
            nl, mode, load, rng, lat, p1, p99, thr = future.result()
            if lat is None or p1 is None or p99 is None or thr is None:
                failed.append((nl, mode, load, rng))
            if completed % 50 == 0:
                print(f"  [{completed}/{len(jobs)}] completed", flush=True)

    print(f"[*] Parallel pass complete: {len(jobs) - len(failed)}/{len(jobs)} successful", 
          flush=True)

    # Retry failed jobs
    if RERUN_MISSING and failed:
        print(f"[*] Re-running {len(failed)} failed jobs sequentially...", flush=True)
        for i, (nl, mode, load, rng) in enumerate(failed):
            run_job(nl, mode, load, rng)
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(failed)}] re-run complete", flush=True)

    refresh_plot_data()
    raw = load_existing_results(require_p99=True)
    plot_results(raw)

    print("[*] All jobs complete!", flush=True)

if __name__ == "__main__":
    main()
