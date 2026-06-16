"""
Runner + plotter for scratch/4.1.4/wifi-mlo-throughput-scenario1-yans
Throughput Scenario #3: main BSS throughput vs. OBSS station count (0-10).
Modes: SLO, STR, EMLSR  |  5 runs each  |  3 x 11 x 5 = 165 simulations.
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
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

# ===== CONFIG =====
NS3_PROGRAM  = "scratch/4.1.4/wifi-mlo-throughput-scenario1-yans"
SIMTIME      = 10.0
PAYLOAD      = 1500
NMDPUS       = 1024
RUNS         = 5
RNG_START    = 1001
CI           = 0.95
MAX_WORKERS  = os.cpu_count() or 8
OUTPUT_DIR   = "thr_runs_4_1_4"
PLOT_DIR     = "plots"
PLOT_FILE    = os.path.join(PLOT_DIR, "4_1_4.svg")
MAX_RETRIES  = 3     # max rerun attempts on transient error
RERUN_MISSING = True  # after all jobs finish, rerun any that failed

MODES        = ["SLO", "STR", "EMLSR"]
OBSS_COUNTS  = list(range(11))   # 0 .. 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RE_DL = re.compile(r"MainBSS DL MacRx:\s*([0-9.]+)\s*Mb/s")
RE_UL = re.compile(r"MainBSS UL MacRx:\s*([0-9.]+)\s*Mb/s")


# ---------------------------------------------------------------------------

def _run_once(mode, n_obss, rng_run):
    """Single attempt: return (dl, ul) in Mb/s or raise RuntimeError."""
    args = (
        f"{NS3_PROGRAM} "
        f"--RngRun={rng_run} "
        f"--simTime={SIMTIME} "
        f"--payloadSize={PAYLOAD} "
        f"--nMpdus={NMDPUS} "
        f"--mloMode={mode} "
        f"--nObss={n_obss}"
    )
    cmd = ["./ns3", "run", args]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log = os.path.join(OUTPUT_DIR, f"{mode}_n{n_obss}_rng{rng_run}_err.txt")
        with open(log, "w") as f:
            f.write("CMD:\n" + " ".join(cmd)
                    + "\n\nSTDOUT:\n" + result.stdout
                    + "\n\nSTDERR:\n" + result.stderr)
        raise RuntimeError(f"exit={result.returncode}, log={log}")

    dl = ul = None
    for line in result.stdout.splitlines():
        if dl is None:
            m = RE_DL.search(line)
            if m:
                dl = float(m.group(1))
        if ul is None:
            m = RE_UL.search(line)
            if m:
                ul = float(m.group(1))

    if dl is None or ul is None:
        log = os.path.join(OUTPUT_DIR, f"{mode}_n{n_obss}_rng{rng_run}_stdout.txt")
        with open(log, "w") as f:
            f.write(result.stdout)
        raise RuntimeError(f"DL or UL line not found (dl={dl}, ul={ul}); log={log}")

    return dl, ul


def run_one(mode, n_obss, rng_run):
    """Return (dl_thr, ul_thr) in Mb/s; retries up to MAX_RETRIES times on error."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _run_once(mode, n_obss, rng_run)
        except RuntimeError as e:
            last_exc = e
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {mode} nObss={n_obss} "
                  f"rng{rng_run}: {e}")
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_exc}")


def ci_hw(data, confidence=CI):
    n = len(data)
    if n < 2:
        return float(np.mean(data)), 0.0
    mean = float(np.mean(data))
    h    = float(stats.sem(data) * stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return mean, h


# ---------------------------------------------------------------------------

def main():
    # Build job list: (mode, n_obss, rng_run)
    jobs = [(mode, n, rr)
            for mode in MODES
            for n    in OBSS_COUNTS
            for rr   in range(RNG_START, RNG_START + RUNS)]

    total = len(jobs)
    print(f"Launching {total} simulations on {MAX_WORKERS} workers ...\n")

    # raw[mode][direction][n_obss] = list of per-run throughputs
    DIRS = ["DL", "UL"]
    raw  = {m: {d: {n: [] for n in OBSS_COUNTS} for d in DIRS} for m in MODES}
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut = {ex.submit(run_one, mode, n, rr): (mode, n, rr)
               for mode, n, rr in jobs}
        for future in as_completed(fut):
            mode, n, rr = fut[future]
            done += 1
            try:
                dl, ul = future.result()
                raw[mode]["DL"][n].append(dl)
                raw[mode]["UL"][n].append(ul)
                print(f"  [{done:>3}/{total}]  {mode:<5}  nObss={n:>2}  "
                      f"RngRun={rr}  DL={dl:.1f}  UL={ul:.1f} Mb/s")
            except Exception as e:
                print(f"  [{done:>3}/{total}]  [ERROR] {mode} nObss={n} rng{rr}: {e}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    means = {m: {d: [] for d in DIRS} for m in MODES}
    hw    = {m: {d: [] for d in DIRS} for m in MODES}
    for m in MODES:
        print(f"\n{m}:")
        for n in OBSS_COUNTS:
            for d in DIRS:
                mn, h = ci_hw(raw[m][d][n])
                means[m][d].append(mn)
                hw[m][d].append(h)
            print(f"  nObss={n:>2}: "
                  f"DL {means[m]['DL'][-1]:.2f}±{hw[m]['DL'][-1]:.2f}  "
                  f"UL {means[m]['UL'][-1]:.2f}±{hw[m]['UL'][-1]:.2f} Mb/s  "
                  f"(n={len(raw[m]['DL'][n])})")

    # ── Plot ───────────────────────────────────────────────────────────────
    colors     = {"SLO": "#4C72B0", "STR": "#55A868", "EMLSR": "#DD8452"}
    markers    = {"SLO": "o",       "STR": "s",        "EMLSR": "^"}
    linestyles = {"DL": "-",        "UL": "--"}
    x          = np.array(OBSS_COUNTS)

    fig, ax = plt.subplots(figsize=(11, 6))

    for m in MODES:
        for d in DIRS:
            mn_arr = np.array(means[m][d])
            hw_arr = np.array(hw[m][d])
            ax.errorbar(
                x, mn_arr, yerr=hw_arr,
                label=f"{m} {d}",
                color=colors[m],
                linestyle=linestyles[d],
                marker=markers[m],
                markersize=6,
                linewidth=2,
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
                zorder=3,
            )

    ax.set_xlabel("Number of OBSS stations per link", fontsize=12)
    ax.set_ylabel("Main BSS Throughput [Mbit/s]", fontsize=12)
    ax.set_xticks(x)
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Scenario 3 \u2013 Main BSS throughput vs. OBSS load\n"
        f"5 GHz (ch42) + 6 GHz (ch7), 80\u202fMHz, EHT MCS\u202f13  |  "
        f"Mean \u00b1 {int(CI*100)}% CI  |  {RUNS} runs",
        fontsize=12,
    )
    ax.legend(title="Mode / Direction", fontsize=10, title_fontsize=10,
              ncol=2, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    plt.show()
    print(f"\n[OK] Plot saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()

