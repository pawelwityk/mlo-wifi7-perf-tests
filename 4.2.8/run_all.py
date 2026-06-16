"""
Runner for scratch/4.2.8/wifi-mlo-emlsr-params
EMLSR Parameter Variation — Throughput & Latency

Sweeps each EMLSR parameter individually while keeping others at baseline.
Each sweep point: RUNS × nStas combos.

Baseline: transDelayUs=0, padDelayUs=0, timeoutUs=128, auxWidth=80,
          sleep=false, txCap=true, switchAux=true, nStas=1, load=0.7
"""

# >>> thesis-style shim
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401
# <<< thesis-style shim
import subprocess
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== CONFIG =====
NS3_BINARY    = "build/scratch/4.2.8/ns3.45-wifi-mlo-emlsr-params-optimized"
SIMTIME       = 3.0      # short window; channel-access-delay converges quickly
PAYLOAD       = 1500
NMDPUS        = 1024
RUNS          = 5        # fewer reps for fast turnaround; CI tightens with sqrt(N)
RNG_START     = 1001
MAX_WORKERS   = 11
OUTPUT_DIR    = "thr_runs_4_2_8"
MAX_RETRIES   = 3
RERUN_MISSING = True

# Fixed load for parameter variation (moderate load where differences show)
NORMALIZED_LOAD = 0.7
NSTA_VALUES     = [1, 4]

# Parameter sweeps (each varied independently, others at baseline)
SWEEPS = {
    "transDelay": [0, 16, 32, 64, 128, 256],          # µs
    "padDelay":   [0, 32, 64, 128, 256],               # µs
    "timeout":    [128, 256, 512, 1024, 2048, 4096, 8192, 16384],  # µs
    "auxWidth":   [20, 40, 80],                         # MHz
    "sleep":      [0, 1],                               # bool
    "txCap":      [0, 1],                               # bool
    "switchAux":  [0, 1],                               # bool
}

# Baseline values
BASELINE = {
    "transDelay": 0,
    "padDelay": 0,
    "timeout": 128,
    "auxWidth": 80,
    "sleep": 0,
    "txCap": 1,
    "switchAux": 1,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

RE_LAT = re.compile(r"Mean DL Latency:\s*([0-9.]+)\s*ms")
RE_THR = re.compile(r"Mean DL Throughput:\s*([0-9.]+)\s*Mb/s")


def _key(param, val, ns):
    return f"P{param}_V{val}_S{ns}"


def _result_path(param, val, ns, rng_run):
    return os.path.join(OUTPUT_DIR, f"{_key(param, val, ns)}_rng{rng_run}_result.csv")


def save_result(param, val, ns, rng_run, lat, thr):
    with open(_result_path(param, val, ns, rng_run), "w") as f:
        f.write(f"{lat},{thr}\n")


def load_existing_results():
    done_set = set()
    pattern = re.compile(r"^P(\w+)_V([^_]+)_S(\d+)_rng(\d+)_result\.csv$")
    if not os.path.isdir(OUTPUT_DIR):
        return done_set
    for fname in os.listdir(OUTPUT_DIR):
        m = pattern.match(fname)
        if not m:
            continue
        param, val_s, ns_s, rr_s = m.group(1), m.group(2), m.group(3), m.group(4)
        done_set.add((param, val_s, int(ns_s), int(rr_s)))
    return done_set


def _build_args(param, val, ns):
    """Build command-line args with baseline + the one swept parameter."""
    p = dict(BASELINE)
    p[param] = val

    return [
        NS3_BINARY,
        f"--simTime={SIMTIME}",
        f"--payloadSize={PAYLOAD}",
        f"--nMpdus={NMDPUS}",
        f"--nStas={ns}",
        f"--normalizedLoad={NORMALIZED_LOAD}",
        f"--transitionDelayUs={p['transDelay']}",
        f"--paddingDelayUs={p['padDelay']}",
        f"--transitionTimeoutUs={p['timeout']}",
        f"--auxPhyWidth={p['auxWidth']}",
        f"--putAuxPhyToSleep={'true' if p['sleep'] else 'false'}",
        f"--auxPhyTxCapable={'true' if p['txCap'] else 'false'}",
        f"--switchAuxPhy={'true' if p['switchAux'] else 'false'}",
    ]


def _run_once(param, val, ns, rng_run):
    cmd = _build_args(param, val, ns) + [f"--RngRun={rng_run}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        log = os.path.join(OUTPUT_DIR, f"{_key(param, val, ns)}_rng{rng_run}_err.txt")
        with open(log, "w") as f:
            f.write("CMD:\n" + " ".join(cmd)
                    + "\n\nSTDOUT:\n" + result.stdout
                    + "\n\nSTDERR:\n" + result.stderr)
        raise RuntimeError(f"exit={result.returncode}, log={log}")

    lat = thr = None
    for line in result.stdout.splitlines():
        if lat is None:
            m = RE_LAT.search(line)
            if m:
                lat = float(m.group(1))
        if thr is None:
            m = RE_THR.search(line)
            if m:
                thr = float(m.group(1))

    if lat is None or thr is None:
        log = os.path.join(OUTPUT_DIR, f"{_key(param, val, ns)}_rng{rng_run}_stdout.txt")
        with open(log, "w") as f:
            f.write(result.stdout)
        raise RuntimeError(f"Latency or throughput not parsed; log={log}")

    return lat, thr


def run_one(param, val, ns, rng_run):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _run_once(param, val, ns, rng_run)
        except RuntimeError as e:
            last_exc = e
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {_key(param, val, ns)} rng{rng_run}: {e}")
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_exc}")


def main():
    # Generate all jobs
    jobs = []
    for param, values in SWEEPS.items():
        for val in values:
            for ns in NSTA_VALUES:
                for rr in range(RNG_START, RNG_START + RUNS):
                    jobs.append((param, val, ns, rr))

    already_done = load_existing_results()
    pending_jobs = [j for j in jobs if (j[0], str(j[1]), j[2], j[3]) not in already_done]
    skipped = len(jobs) - len(pending_jobs)

    if already_done:
        print(f"  Loaded {len(already_done)} result(s) from '{OUTPUT_DIR}'.")
    if skipped:
        print(f"  Skipping {skipped} already-completed job(s).\n")

    failed_jobs = []
    done = 0

    if not pending_jobs:
        print("  All jobs already completed.\n")
    else:
        print(f"Launching {len(pending_jobs)} simulation(s) on {MAX_WORKERS} workers ...\n")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_one, param, val, ns, rr): (param, val, ns, rr)
                for param, val, ns, rr in pending_jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                param, val, ns, rr = job
                try:
                    lat, thr = future.result()
                    save_result(param, val, ns, rr, lat, thr)
                    done += 1
                    pct = int(100 * done / len(pending_jobs))
                    print(f"[{pct:3d}% {done:3d}/{len(pending_jobs)}] "
                          f"{_key(param, val, ns)} rng{rr}: {lat:.3f} ms, {thr:.1f} Mb/s")
                except RuntimeError as e:
                    failed_jobs.append((job, str(e)))
                    print(f"[FAIL] {job}: {e}")

    if failed_jobs and RERUN_MISSING:
        print(f"\n[!] {len(failed_jobs)} job(s) failed — retrying sequentially ...\n")
        retry_jobs = [j for j, _ in failed_jobs]
        for param, val, ns, rr in retry_jobs:
            try:
                lat, thr = run_one(param, val, ns, rr)
                save_result(param, val, ns, rr, lat, thr)
                print(f"[RETRY OK] {_key(param, val, ns)} rng{rr}: {lat:.3f} ms, {thr:.1f} Mb/s")
                failed_jobs = [(j, e) for j, e in failed_jobs if j != (param, val, ns, rr)]
            except RuntimeError as e:
                print(f"[RETRY FAIL] {_key(param, val, ns)} rng{rr}: {e}")

    if failed_jobs:
        print(f"\n[!] {len(failed_jobs)} job(s) still failing:")
        for (param, val, ns, rr), exc in failed_jobs:
            print(f"    {_key(param, val, ns)} rng{rr}: {exc}")
    else:
        print("\n[OK] All jobs completed!")

    print(f"\n[Next] python3 scratch/4.2.8/plot_emlsr_params.py")


if __name__ == "__main__":
    main()
