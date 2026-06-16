"""
Runner for scratch/4.2.7/wifi-mlo-emlsr-vs-slo
EMLSR vs SLO — Throughput & Latency under varying normalized load

Sweep: mloMode ∈ {SLO, EMLSR} × nStas ∈ {1, 4, 10}
       × normalizedLoad ∈ {0.0, 0.1, ..., 1.0}
       = 2 × 3 × 11 = 66 configs × 10 runs = 660 simulations

Reference: 80 MHz single-link peak = 1114 Mb/s (nMpdus=1024)
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
NS3_BINARY    = "build/scratch/4.2.7/ns3.45-wifi-mlo-emlsr-vs-slo-optimized"
SIMTIME       = 5.0      # 5 s sufficient for channel-access-delay measurement
PAYLOAD       = 1500
NMDPUS        = 1024
RUNS          = 10
RNG_START     = 1001
MAX_WORKERS   = 11
OUTPUT_DIR    = "thr_runs_4_2_7"
MAX_RETRIES   = 3
RERUN_MISSING = True

MLO_MODES     = ["SLO", "EMLSR"]
NSTA_VALUES   = [1, 4, 10]
NORMALIZED_LOADS = [round(0.1 * i, 1) for i in range(0, 11)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

RE_LAT = re.compile(r"Mean DL Latency:\s*([0-9.]+)\s*ms")
RE_THR = re.compile(r"Mean DL Throughput:\s*([0-9.]+)\s*Mb/s")


def _key(mode, ns, rho):
    rho_int = int(round(rho * 10))
    return f"M{mode}_S{ns}_R{rho_int}"


def _result_path(mode, ns, rho, rng_run):
    return os.path.join(OUTPUT_DIR, f"{_key(mode, ns, rho)}_rng{rng_run}_result.csv")


def save_result(mode, ns, rho, rng_run, lat, thr):
    with open(_result_path(mode, ns, rho, rng_run), "w") as f:
        f.write(f"{lat},{thr}\n")


def load_existing_results():
    done_set = set()
    pattern = re.compile(r"^M(SLO|EMLSR)_S(\d+)_R(\d+)_rng(\d+)_result\.csv$")
    if not os.path.isdir(OUTPUT_DIR):
        return done_set
    for fname in os.listdir(OUTPUT_DIR):
        m = pattern.match(fname)
        if not m:
            continue
        mode = m.group(1)
        ns, rho_int, rr = int(m.group(2)), int(m.group(3)), int(m.group(4))
        rho = rho_int / 10.0
        done_set.add((mode, ns, rho, rr))
    return done_set


def _run_once(mode, ns, rho, rng_run):
    cmd = [
        NS3_BINARY,
        f"--RngRun={rng_run}",
        f"--simTime={SIMTIME}",
        f"--payloadSize={PAYLOAD}",
        f"--nMpdus={NMDPUS}",
        f"--mloMode={mode}",
        f"--nStas={ns}",
        f"--normalizedLoad={rho}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

    if result.returncode != 0:
        log = os.path.join(OUTPUT_DIR, f"{_key(mode, ns, rho)}_rng{rng_run}_err.txt")
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
        log = os.path.join(OUTPUT_DIR, f"{_key(mode, ns, rho)}_rng{rng_run}_stdout.txt")
        with open(log, "w") as f:
            f.write(result.stdout)
        raise RuntimeError(f"Latency or throughput not parsed; log={log}")

    return lat, thr


def run_one(mode, ns, rho, rng_run):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _run_once(mode, ns, rho, rng_run)
        except RuntimeError as e:
            last_exc = e
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {_key(mode, ns, rho)} rng{rng_run}: {e}")
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_exc}")


def main():
    jobs = [(mode, ns, rho, rr)
            for mode in MLO_MODES
            for ns in NSTA_VALUES
            for rho in NORMALIZED_LOADS
            for rr in range(RNG_START, RNG_START + RUNS)]

    already_done = load_existing_results()
    pending_jobs = [j for j in jobs if j not in already_done]
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
                executor.submit(run_one, mode, ns, rho, rr): (mode, ns, rho, rr)
                for mode, ns, rho, rr in pending_jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                mode, ns, rho, rr = job
                try:
                    lat, thr = future.result()
                    save_result(mode, ns, rho, rr, lat, thr)
                    done += 1
                    pct = int(100 * done / len(pending_jobs))
                    print(f"[{pct:3d}% {done:3d}/{len(pending_jobs)}] "
                          f"{_key(mode, ns, rho)} rng{rr}: {lat:.2f} ms, {thr:.1f} Mb/s")
                except RuntimeError as e:
                    failed_jobs.append((job, str(e)))
                    print(f"[FAIL] {job}: {e}")

    if failed_jobs and RERUN_MISSING:
        print(f"\n[!] {len(failed_jobs)} job(s) failed — retrying sequentially ...\n")
        retry_jobs = [j for j, _ in failed_jobs]
        for mode, ns, rho, rr in retry_jobs:
            try:
                lat, thr = run_one(mode, ns, rho, rr)
                save_result(mode, ns, rho, rr, lat, thr)
                print(f"[RETRY OK] {_key(mode, ns, rho)} rng{rr}: {lat:.2f} ms, {thr:.1f} Mb/s")
                failed_jobs = [(j, e) for j, e in failed_jobs if j != (mode, ns, rho, rr)]
            except RuntimeError as e:
                print(f"[RETRY FAIL] {_key(mode, ns, rho)} rng{rr}: {e}")

    if failed_jobs:
        print(f"\n[!] {len(failed_jobs)} job(s) still failing:")
        for (mode, ns, rho, rr), exc in failed_jobs:
            print(f"    {_key(mode, ns, rho)} rng{rr}: {exc}")
    else:
        print("\n[OK] All jobs completed!")

    print(f"\n[Next] python3 scratch/4.2.7/plot_emlsr_vs_slo.py")


if __name__ == "__main__":
    main()
