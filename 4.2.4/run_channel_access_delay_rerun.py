#!/usr/bin/env python3
"""
run_channel_access_delay_rerun.py -- temporary access-delay sweep for Scenario #4A.

This mirrors the 4.2.4 rerun harness, but parses the experimental MAC-side
channel access delay metric emitted by wifi-mlo-channel-access-scenario4a.cc.
"""

# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim

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

NS3_BINARY = "build/scratch/4.2.4-access-temp/ns3.45-wifi-mlo-latency-scenario4a-access-temp-optimized"
OUTPUT_DIR = "thr_runs_4_2_4_access_delay"
PLOT_DIR = "plots"
PLOT_FILE = os.path.join(PLOT_DIR, "4_2_4_access_delay.svg")

NUM_LINKS = [1, 2, 4]
OFFERED_LOADS = list(range(100, 2600, 100))
RUNS = 10
SIM_TIME = 3.0
PAYLOAD = 1500
NMDPUS = 1024
STARTUP_GUARD = 0.3
MAX_WORKERS = 11
RERUN_MISSING = True

T95_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RE_DELAY = re.compile(r"Mean DL Channel Access Delay:\s*([0-9.]+)\s*ms")
RE_P50 = re.compile(r"DL Channel Access Delay p50:\s*([0-9.]+)\s*ms")
RE_P95 = re.compile(r"DL Channel Access Delay p95:\s*([0-9.]+)\s*ms")
RE_P99 = re.compile(r"DL Channel Access Delay p99:\s*([0-9.]+)\s*ms")
RE_THR = re.compile(r"Mean DL Throughput:\s*([0-9.]+)\s*Mb/s")

MODE_LABELS = {1: "SLO (1 link)", 2: "STR2 (2 links)", 4: "STR4 (4 links)"}


def _result_path(nl: int, load: int, rng: int) -> str:
    return os.path.join(OUTPUT_DIR, f"L{nl}_Load{load}_rng{rng}_result.csv")


def _key(nl: int, load: int) -> tuple:
    return (nl, load)


def _parse_result_row(text: str):
    reader = csv.reader([text])
    row = next(reader, [])
    parts = [p.strip() for p in row if p.strip()]
    if len(parts) >= 5:
        return tuple(float(p) for p in parts[:5])
    if len(parts) >= 4:
        delay, p50, p99, thr = (float(p) for p in parts[:4])
        return delay, p50, p99, p99, thr
    if len(parts) >= 2:
        delay, thr = float(parts[0]), float(parts[1])
        return delay, delay, delay, delay, thr
    return None


def run_job(nl: int, load: int, rng: int, force_refresh: bool = False) -> tuple:
    result_file = _result_path(nl, load, rng)
    if os.path.exists(result_file) and not force_refresh:
        try:
            with open(result_file) as f:
                parsed = _parse_result_row(f.read().strip())
            if parsed is not None:
                delay, p50, p95, p99, thr = parsed
                return (nl, load, rng, delay, p50, p95, p99, thr)
        except Exception:
            pass

    cmd = [
        NS3_BINARY,
        f"--simTime={SIM_TIME}",
        f"--payloadSize={PAYLOAD}",
        f"--nMpdus={NMDPUS}",
        f"--numLinks={nl}",
        f"--offeredLoad={load}",
        f"--startupGuard={STARTUP_GUARD}",
        f"--RngRun={rng}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        delay_m = RE_DELAY.search(result.stdout)
        p50_m = RE_P50.search(result.stdout)
        p95_m = RE_P95.search(result.stdout)
        p99_m = RE_P99.search(result.stdout)
        thr_m = RE_THR.search(result.stdout)
        if delay_m and thr_m:
            delay = float(delay_m.group(1))
            p50 = float(p50_m.group(1)) if p50_m else delay
            p95 = float(p95_m.group(1)) if p95_m else delay
            p99 = float(p99_m.group(1)) if p99_m else delay
            thr = float(thr_m.group(1))
            with open(result_file, "w") as f:
                f.write(f"{delay},{p50},{p95},{p99},{thr}\n")
            return (nl, load, rng, delay, p50, p95, p99, thr)
        print(f"  PARSE-FAIL L{nl} Load{load} rng{rng}", file=sys.stderr)
        print(f"  stdout: {result.stdout[-400:]}", file=sys.stderr)
    except Exception as e:
        print(f"  ERROR L{nl} Load{load} rng{rng}: {e}", file=sys.stderr)

    return (nl, load, rng, None, None, None, None, None)


def load_existing_results():
    results = {}
    if not os.path.isdir(OUTPUT_DIR):
        return results
    for fname in os.listdir(OUTPUT_DIR):
        m = re.match(r"L(\d+)_Load(\d+)_rng(\d+)_result\.csv$", fname)
        if not m:
            continue
        nl, load, rng = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if nl not in NUM_LINKS:
            continue
        fpath = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(fpath) as f:
                parsed = _parse_result_row(f.read().strip())
            if parsed is None:
                continue
            delay, p50, p95, p99, thr = parsed
            results.setdefault(_key(nl, load), []).append((rng, delay, p50, p95, p99, thr))
        except Exception:
            pass
    return results


def ci95(values):
    from _robust import ci_hw_robust
    return ci_hw_robust(values, confidence=0.95)[1]


def _filtered(values):
    from _robust import remove_outliers_robust
    return remove_outliers_robust(values)


def plot_results(raw: dict) -> None:
    colors = {1: "#2E75D8", 2: "#E63B0E", 4: "#28A745"}
    markers = {1: "o", 2: "s", 4: "^"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Channel Access Delay Scenario #4A")

    for nl in NUM_LINKS:
        xs, delay_m, delay_e, p99_m, p99_e, thr_m, thr_e = [], [], [], [], [], [], []
        for load in OFFERED_LOADS:
            rows = raw.get(_key(nl, load), [])
            if not rows:
                continue
            delays = _filtered([r[1] for r in rows])
            p99s = _filtered([r[4] for r in rows])
            thrs = _filtered([r[5] for r in rows])
            xs.append(load)
            delay_m.append(mean(delays))
            delay_e.append(ci95(delays))
            p99_m.append(mean(p99s))
            p99_e.append(ci95(p99s))
            thr_m.append(mean(thrs))
            thr_e.append(ci95(thrs))

        if not xs:
            continue

        c, mk = colors[nl], markers[nl]
        axes[0].errorbar(xs, delay_m, yerr=delay_e, label=f"{MODE_LABELS[nl]} mean", color=c, marker=mk, linestyle="-", markersize=4, capsize=3, linewidth=1.6)
        axes[0].errorbar(xs, p99_m, yerr=p99_e, label=f"{MODE_LABELS[nl]} p99", color=c, marker=mk, linestyle="dotted", markersize=4, capsize=3, linewidth=1.2)
        axes[1].errorbar(xs, thr_m, yerr=thr_e, label=MODE_LABELS[nl], color=c, marker=mk, linestyle="-", markersize=4, capsize=3, linewidth=1.6)

    for ax, ylabel in zip(axes, ["Channel Access Delay (ms)", "Aggregate Throughput [Mbit/s]"]):
        ax.set_xlabel("Total Offered Load [Mbit/s]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    print(f"\n[plot] saved -> {PLOT_FILE}", flush=True)
    plt.close(fig)


def main():
    print("[*] Loading existing results...", flush=True)
    existing = load_existing_results()
    loaded = sum(len(v) for v in existing.values())
    print(f"[*] Found {loaded} existing run results", flush=True)

    print(f"[*] Scheduling jobs: numLinks={NUM_LINKS}, loads={len(OFFERED_LOADS)} points, {RUNS} runs each", flush=True)
    jobs = list(product(NUM_LINKS, OFFERED_LOADS, range(1, RUNS + 1)))
    print(f"[*] Total jobs: {len(jobs)}", flush=True)

    failed = []
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_job, nl, load, rng): (nl, load, rng) for nl, load, rng in jobs}
        for future in as_completed(futures):
            nl, load, rng = futures[future]
            try:
                res = future.result()
                if res[3] is not None:
                    existing.setdefault(_key(nl, load), []).append((rng, *res[3:]))
                    completed += 1
                    if completed % 50 == 0:
                        print(f"[*] Completed {completed}/{len(jobs)}", flush=True)
                else:
                    failed.append((nl, load, rng))
            except Exception as e:
                print(f"ERROR in job L{nl} Load{load} rng{rng}: {e}", flush=True)
                failed.append((nl, load, rng))

    if RERUN_MISSING and failed:
        print(f"\n[*] Re-running {len(failed)} failed jobs sequentially...", flush=True)
        for nl, load, rng in failed:
            res = run_job(nl, load, rng, force_refresh=True)
            if res[3] is not None:
                existing.setdefault(_key(nl, load), []).append((rng, *res[3:]))
            else:
                print(f"  STILL FAILED: L{nl} Load{load} rng{rng}", flush=True)

    print("\n[*] Generating plot...", flush=True)
    plot_results(existing)
    print("[*] Done.", flush=True)


if __name__ == "__main__":
    main()
