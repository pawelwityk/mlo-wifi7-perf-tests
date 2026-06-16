"""
Runner for scratch/4.2.1/wifi-mlo-latency-scenario1
Latency Scenario #1 (based on [16] – Jeknic & Kocan, IEEE IT 2024)

Sweep: numLinks ∈ {1, 2} × channelWidth ∈ {80, 160} × nStas ∈ {1, 4, 10} 
       × normalizedLoad ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
       = 72 configurations × 10 runs = 720 simulations.

Normalized Load Reference (per-user SLO max, single-link, nMpdus=1024):
  - 80 MHz:  1114 Mb/s  → rho=1.0 means 1114 Mb/s offered total
  - 160 MHz: 1279 Mb/s  → rho=1.0 means 1279 Mb/s offered total
    Measured from 4.1.1 baseline (mean of 20 runs). nMpdus=1024 required
    in ns-3 to avoid queue starvation at 160 MHz PHY rates.

Features:
  - Per-run results persisted to OUTPUT_DIR/*_result.csv — script resumes
    automatically after a crash.
  - RERUN_MISSING: retries failed jobs sequentially after the parallel sweep.
  - Uses rho-tagged filenames: L{nl}_W{cw}_S{ns}_R{rho_int}_rng{rr}_result.csv
"""

# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim
import subprocess
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== CONFIG =====
NS3_BINARY    = "build/scratch/4.2.1/ns3.45-wifi-mlo-latency-scenario1-optimized"
SIMTIME       = 5.0      # 5 s sufficient for channel-access-delay measurement
PAYLOAD       = 1500
NMDPUS        = 1024     # A-MPDU size (was 64; raised to remove 160 MHz cap)
RUNS          = 10
RNG_START     = 1001
MAX_WORKERS   = 6
OUTPUT_DIR    = "thr_runs_4_2_1"
MAX_RETRIES   = 3
RERUN_MISSING = True

NUM_LINKS     = [1, 2]
CHAN_WIDTHS   = [80, 160]   # MHz
NSTA_VALUES   = [1, 4, 10]
NORMALIZED_LOADS = [round(0.1 * i, 1) for i in range(0, 11)]  # 0.0..1.0 step 0.1 (densified to match paper Fig 2 smoothness)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RE_LAT = re.compile(r"Mean DL Latency:\s*([0-9.]+)\s*ms")
RE_THR = re.compile(r"Mean DL Throughput:\s*([0-9.]+)\s*Mb/s")
RE_SAT = re.compile(r"^SATURATED:")


# ---------------------------------------------------------------------------
# Per-run result persistence helpers

def _key(nl, cw, ns, rho):
    """Generate result file key with rho in tenths (0.0 → R0, 0.6 → R6, 1.0 → R10)."""
    rho_int = int(round(rho * 10))
    return f"L{nl}_W{cw}_S{ns}_R{rho_int}"


def _result_path(nl, cw, ns, rho, rng_run):
    """Result file: L{nl}_W{cw}_S{ns}_R{rho_int}_rng{rng_run}_result.csv"""
    return os.path.join(OUTPUT_DIR, f"{_key(nl, cw, ns, rho)}_rng{rng_run}_result.csv")


def _sat_path(nl, cw, ns, rho, rng_run):
    """Sentinel file written when a run is rejected as saturated."""
    return os.path.join(OUTPUT_DIR, f"{_key(nl, cw, ns, rho)}_rng{rng_run}_saturated")


def save_result(nl, cw, ns, rho, rng_run, lat, thr):
    with open(_result_path(nl, cw, ns, rho, rng_run), "w") as f:
        f.write(f"{lat},{thr}\n")


def save_saturated(nl, cw, ns, rho, rng_run):
    open(_sat_path(nl, cw, ns, rho, rng_run), "w").close()


def load_existing_results():
    """Return set of completed (nl, cw, ns, rho, rr) tuples (results + saturated sentinels)."""
    done_set = set()
    pat_result = re.compile(r"^L(\d+)_W(\d+)_S(\d+)_R(\d+)_rng(\d+)_result\.csv$")
    pat_sat    = re.compile(r"^L(\d+)_W(\d+)_S(\d+)_R(\d+)_rng(\d+)_saturated$")
    if not os.path.isdir(OUTPUT_DIR):
        return done_set
    for fname in os.listdir(OUTPUT_DIR):
        m = pat_result.match(fname) or pat_sat.match(fname)
        if not m:
            continue
        nl, cw, ns, rho_int, rr = map(int, m.groups())
        if nl not in NUM_LINKS or cw not in CHAN_WIDTHS or ns not in NSTA_VALUES:
            continue
        rho = rho_int / 10.0
        if abs(rho - round(rho, 1)) > 1e-6:
            continue
        done_set.add((nl, cw, ns, rho, rr))
    return done_set


# ---------------------------------------------------------------------------

def _run_once(nl, cw, ns, rho, rng_run):
    """Single attempt: return (latency_ms, throughput_mbps) or raise RuntimeError."""
    cmd = [
        NS3_BINARY,
        f"--RngRun={rng_run}",
        f"--simTime={SIMTIME}",
        f"--payloadSize={PAYLOAD}",
        f"--nMpdus={NMDPUS}",
        f"--numLinks={nl}",
        f"--channelWidth={cw}",
        f"--nStas={ns}",
        f"--normalizedLoad={rho}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        log = os.path.join(OUTPUT_DIR, f"{_key(nl, cw, ns, rho)}_rng{rng_run}_err.txt")
        with open(log, "w") as f:
            f.write("CMD:\n" + " ".join(cmd)
                    + "\n\nSTDOUT:\n" + result.stdout
                    + "\n\nSTDERR:\n" + result.stderr)
        raise RuntimeError(f"exit={result.returncode}, log={log}")

    saturated = any(RE_SAT.search(ln) for ln in result.stdout.splitlines())

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
        log = os.path.join(OUTPUT_DIR, f"{_key(nl, cw, ns, rho)}_rng{rng_run}_stdout.txt")
        with open(log, "w") as f:
            f.write(result.stdout)
        raise RuntimeError(f"Latency or throughput line not found; log={log}")

    return lat, thr, saturated


def run_one(nl, cw, ns, rho, rng_run):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _run_once(nl, cw, ns, rho, rng_run)
        except RuntimeError as e:
            last_exc = e
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] L{nl} {cw}MHz S{ns} "
                  f"rho={rho} rng{rng_run}: {e}")
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_exc}")



# ---------------------------------------------------------------------------

def main():
    """Orchestrate parallel sweep of all jobs."""
    # Generate all jobs: (nl, cw, ns, rho, rng_run)
    jobs = [(nl, cw, ns, rho, rr)
            for nl in NUM_LINKS
            for cw in CHAN_WIDTHS
            for ns in NSTA_VALUES
            for rho in NORMALIZED_LOADS
            for rr in range(RNG_START, RNG_START + RUNS)]

    failed_jobs = []
    done = 0

    # ── Load results from any previous run ────────────────────────────────
    already_done = load_existing_results()
    if already_done:
        print(f"  Loaded {len(already_done)} result(s) from previous run(s) in '{OUTPUT_DIR}'.")
    
    pending_jobs = [j for j in jobs if j not in already_done]
    skipped = len(jobs) - len(pending_jobs)
    if skipped:
        print(f"  Skipping {skipped} already-completed job(s).\n")

    if not pending_jobs:
        print("  All jobs already completed — skipping simulation phase.\n")
    else:
        print(f"Launching {len(pending_jobs)} simulation(s) on {MAX_WORKERS} workers ...\n")

        # ── Parallel sweep ────────────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_one, nl, cw, ns, rho, rr): (nl, cw, ns, rho, rr)
                for nl, cw, ns, rho, rr in pending_jobs
            }

            for future in as_completed(futures):
                job = futures[future]
                nl, cw, ns, rho, rr = job
                try:
                    lat, thr, saturated = future.result()
                    done += 1
                    pct = int(100 * done / len(pending_jobs))
                    save_result(nl, cw, ns, rho, rr, lat, thr)
                    if saturated:
                        save_saturated(nl, cw, ns, rho, rr)
                        tag = "SAT"
                    else:
                        tag = "ok "
                    print(f"[{pct:3d}% {done:3d}/{len(pending_jobs)}] {tag} "
                          f"L{nl} W{cw} S{ns} R{int(rho*10)} rng{rr}: "
                          f"{lat:.2f} ms, {thr:.1f} Mb/s")
                except RuntimeError as e:
                    failed_jobs.append((job, str(e)))
                    print(f"[FAIL] {job}: {e}")

    # ── Retry failed jobs sequentially ────────────────────────────────────
    if failed_jobs and RERUN_MISSING:
        print(f"\n[!] {len(failed_jobs)} job(s) failed in parallel phase.")
        print("    Retrying sequentially ...\n")
        retry_jobs = [j for j, _ in failed_jobs]
        for nl, cw, ns, rho, rr in retry_jobs:
            try:
                lat, thr, saturated = run_one(nl, cw, ns, rho, rr)
                save_result(nl, cw, ns, rho, rr, lat, thr)
                if saturated:
                    save_saturated(nl, cw, ns, rho, rr)
                    tag = "SAT"
                else:
                    tag = "ok "
                print(f"[RETRY {tag}] L{nl} W{cw} S{ns} R{int(rho*10)} rng{rr}: "
                      f"{lat:.2f} ms, {thr:.1f} Mb/s")
                failed_jobs = [(j, e) for j, e in failed_jobs if j != (nl, cw, ns, rho, rr)]
            except RuntimeError as e:
                print(f"[RETRY FAIL] L{nl} W{cw} S{ns} R{int(rho*10)} rng{rr}: {e}")

    if failed_jobs:
        print(f"\n[!] {len(failed_jobs)} job(s) still failing after retries:")
        for (nl, cw, ns, rho, rr), exc in failed_jobs:
            print(f"    L{nl} W{cw} S{ns} R{int(rho*10)} rng{rr}: {exc}")
    else:
        print("\n[OK] All jobs completed successfully!")

    # ── Direct user to plot script ────────────────────────────────────────
    print(f"\n[Next] To generate plots, run:")
    print(f"       python3 scratch/4.2.1/plot_delay_vs_load.py")


if __name__ == "__main__":
    main()
