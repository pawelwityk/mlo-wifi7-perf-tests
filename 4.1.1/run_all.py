"""
Runner for scratch/4.1.1/throughput.cc
Scenarios x nMpdus values x 10 RNG runs – fully parallel.
Prints a summary table:  Scenario | nMpdus=64 | nMpdus=1024
"""

# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim
import subprocess
import re
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

# ===== CONFIG =====
NS3_PROGRAM = "scratch/4.1.1/throughput"
SIMTIME     = 10.0
PAYLOAD     = 1500
RUNS        = 10
RNG_START   = 1001
CI          = 0.95
MAX_WORKERS = os.cpu_count() or 8
OUTPUT_DIR  = "thr_runs_4_1_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# (label, mloMode, numLinks, channelWidth_MHz)
SCENARIOS = [
    ("1L  80 MHz",      "SLO", 1, 80),
    ("1L 160 MHz",      "SLO", 1, 160),
    ("2L STR  80 MHz",  "STR", 2, 80),
    ("2L STR 160 MHz",  "STR", 2, 160),
]

NMPDU_VALUES = [64, 1024]

RE_MACRX = re.compile(
    r"MacRx throughput \(STA received bits / simTime\):\s*([0-9.]+)\s*Mb/s"
)


# ---------------------------------------------------------------------------

def run_one(label, mlo_mode, num_links, cw, nMpdus, rng_run):
    args = (
        f"{NS3_PROGRAM} "
        f"--RngRun={rng_run} "
        f"--simTime={SIMTIME} "
        f"--payloadSize={PAYLOAD} "
        f"--nMpdus={nMpdus} "
        f"--numLinks={num_links} "
        f"--mloMode={mlo_mode} "
        f"--channelWidth={cw}"
    )
    cmd = ["./ns3", "run", args]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        safe = f"{label}_n{nMpdus}".replace(" ", "_")
        log = os.path.join(OUTPUT_DIR, f"{safe}_rng{rng_run}_err.txt")
        with open(log, "w") as f:
            f.write("CMD:\n" + " ".join(cmd)
                    + "\n\nSTDOUT:\n" + result.stdout
                    + "\n\nSTDERR:\n" + result.stderr)
        raise RuntimeError(f"exit={result.returncode}, log={log}")

    for line in result.stdout.splitlines():
        m = RE_MACRX.search(line)
        if m:
            return float(m.group(1))

    safe = f"{label}_n{nMpdus}".replace(" ", "_")
    log = os.path.join(OUTPUT_DIR, f"{safe}_rng{rng_run}_stdout.txt")
    with open(log, "w") as f:
        f.write(result.stdout)
    raise RuntimeError(f"MacRx line not found, log={log}")


def ci(data, confidence=CI):
    n = len(data)
    mean = float(np.mean(data))
    if n < 2:
        return mean, 0.0
    h = float(stats.sem(data) * stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return mean, h


# ---------------------------------------------------------------------------

def main():
    # Build full job list
    jobs = [
        (sc_idx, nm_idx, label, mlo_mode, num_links, cw, nMpdus, rr)
        for sc_idx, (label, mlo_mode, num_links, cw) in enumerate(SCENARIOS)
        for nm_idx, nMpdus in enumerate(NMPDU_VALUES)
        for rr in range(RNG_START, RNG_START + RUNS)
    ]

    total = len(jobs)
    print(f"Launching {total} simulations  "
          f"({len(SCENARIOS)} scenarios × {len(NMPDU_VALUES)} nMpdus × {RUNS} runs)  "
          f"on {MAX_WORKERS} workers\n")

    # results[sc_idx][nm_idx] = list of MacRx values
    results = [[[] for _ in NMPDU_VALUES] for _ in SCENARIOS]
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(run_one, label, mlo_mode, num_links, cw, nMpdus, rr):
                (sc_idx, nm_idx, label, nMpdus, rr)
            for sc_idx, nm_idx, label, mlo_mode, num_links, cw, nMpdus, rr in jobs
        }
        for future in as_completed(future_map):
            sc_idx, nm_idx, label, nMpdus, rr = future_map[future]
            done += 1
            try:
                v = future.result()
                results[sc_idx][nm_idx].append(v)
                print(f"  [{done:>3}/{total}]  {label:<18}  "
                      f"nMpdus={nMpdus:<5}  RngRun={rr}  ->  {v:.2f} Mb/s")
            except Exception as e:
                print(f"  [{done:>3}/{total}]  [ERROR]  {label}  "
                      f"nMpdus={nMpdus}  RngRun={rr}: {e}")

    # ------------------------------------------------------------------ table
    COL = 20
    SEP = "─"

    header_scenario = "Scenario"
    headers = [f"nMpdus = {n}" for n in NMPDU_VALUES]

    top    = f"┌{'─'*(COL+2)}┬{'─'*(COL+2)}┬{'─'*(COL+2)}┐"
    mid    = f"├{'─'*(COL+2)}┼{'─'*(COL+2)}┼{'─'*(COL+2)}┤"
    bot    = f"└{'─'*(COL+2)}┴{'─'*(COL+2)}┴{'─'*(COL+2)}┘"

    def row_str(cells):
        return "│ " + " │ ".join(f"{c:<{COL}}" for c in cells) + " │"

    print(f"\n\n{top}")
    print(row_str([header_scenario] + headers))
    print(mid)

    for sc_idx, (label, _, _, _) in enumerate(SCENARIOS):
        cells = [label]
        for nm_idx in range(len(NMPDU_VALUES)):
            vals = results[sc_idx][nm_idx]
            if vals:
                mean, hw = ci(vals)
                cells.append(f"{mean:.1f} ± {hw:.1f} Mb/s")
            else:
                cells.append("N/A")
        print(row_str(cells))

    print(bot)
    print(f"Values: mean ± 95% CI  ({RUNS} independent runs per cell)\n")


if __name__ == "__main__":
    main()
