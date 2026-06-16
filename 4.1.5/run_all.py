"""
Runner + plotter for scratch/4.1.5/wifi-mlo-throughput-scenario1-yans
Throughput Scenario #4: impact of MPDU aggregation and TXOP limit
on STR main BSS DL throughput vs. ch106 occupancy (10-70 %).

Sweep:
  nMpdus      ∈ {64, 512, 1024}
  txopLimitMs ∈ {0, 3}
  ch2OccPct   ∈ {10, 20, 30, 40, 50, 60, 70}   (ch106 occupancy)
  ch1OccPct   ∈ {0, 10}                          (ch42  occupancy)
  -> 3 × 2 × 7 × 2 × RUNS = 420 simulations (5 runs each)

Each (nMpdus, txopLimitMs, ch1OccPct) combination is one subplot /
line group; the x-axis is ch2OccPct.

Output per simulation: "MainBSS DL MacRx: X.XX Mb/s"
"""

# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim
import subprocess
import re
import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

# ===== CONFIG =====
NS3_PROGRAM  = "scratch/4.1.5/wifi-mlo-throughput-scenario1-yans"
SIMTIME      = 10.0
PAYLOAD      = 1500
RUNS         = 5
RNG_START    = 1001
CI           = 0.95
MAX_WORKERS  = os.cpu_count() or 8
OUTPUT_DIR   = "thr_runs_4_1_5"
PLOT_DIR     = "plots"
PLOT_FILE    = os.path.join(PLOT_DIR, "4_1_5.svg")
MAX_RETRIES  = 3

NMDPUS_LIST  = [64, 512, 1024]
TXOP_LIST    = [0.0, 3.0]        # ms
CH2_OCC_LIST = list(range(10, 71, 10))  # 10..70 %
CH1_OCC_LIST = [0.0, 10.0]       # %

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RE_DL = re.compile(r"MainBSS DL MacRx:\s*([0-9.]+)\s*Mb/s")


# ---------------------------------------------------------------------------
# Per-run result persistence (compatible with run_all_rerun.py)

def _result_tag(nm, tx, c2, c1, rr):
    return f"n{nm}_t{int(tx)}_ch2_{int(c2)}_ch1_{int(c1)}_rng{rr}"


def save_result(nm, tx, c2, c1, rr, dl):
    path = os.path.join(OUTPUT_DIR, f"{_result_tag(nm, tx, c2, c1, rr)}_result.csv")
    with open(path, "w") as f:
        f.write(f"{dl}\n")


# ---------------------------------------------------------------------------

def _run_once(nMpdus, txopMs, ch2Occ, ch1Occ, rng_run):
    """Single attempt: return dl_thr [Mbit/s] or raise RuntimeError."""
    args = (
        f"{NS3_PROGRAM} "
        f"--RngRun={rng_run} "
        f"--simTime={SIMTIME} "
        f"--payloadSize={PAYLOAD} "
        f"--nMpdus={nMpdus} "
        f"--txopLimitMs={txopMs} "
        f"--ch2OccPct={ch2Occ} "
        f"--ch1OccPct={ch1Occ}"
    )
    cmd = ["./ns3", "run", args]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        tag = _result_tag(nMpdus, txopMs, ch2Occ, ch1Occ, rng_run)
        log = os.path.join(OUTPUT_DIR, f"{tag}_err.txt")
        with open(log, "w") as f:
            f.write("CMD:\n" + " ".join(cmd)
                    + "\n\nSTDOUT:\n" + result.stdout
                    + "\n\nSTDERR:\n" + result.stderr)
        raise RuntimeError(f"exit={result.returncode}, log={log}")

    for line in result.stdout.splitlines():
        m = RE_DL.search(line)
        if m:
            return float(m.group(1))

    tag = _result_tag(nMpdus, txopMs, ch2Occ, ch1Occ, rng_run)
    log = os.path.join(OUTPUT_DIR, f"{tag}_stdout.txt")
    with open(log, "w") as f:
        f.write(result.stdout)
    raise RuntimeError(f"DL line not found; log={log}")


def run_one(nMpdus, txopMs, ch2Occ, ch1Occ, rng_run):
    """Return dl_thr; retries up to MAX_RETRIES times on error."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _run_once(nMpdus, txopMs, ch2Occ, ch1Occ, rng_run)
        except RuntimeError as e:
            last_exc = e
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] "
                  f"nMpdus={nMpdus} txop={txopMs}ms "
                  f"ch2={ch2Occ}% ch1={ch1Occ}% rng{rng_run}: {e}")
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_exc}")


def ci_hw(data, confidence=CI):
    n = len(data)
    if n < 2:
        return float(np.mean(data)) if data else 0.0, 0.0
    mean = float(np.mean(data))
    h    = float(stats.sem(data) * stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return mean, h


# ---------------------------------------------------------------------------

def main():
    # Build full job list
    keys = list(itertools.product(NMDPUS_LIST, TXOP_LIST, CH2_OCC_LIST, CH1_OCC_LIST))
    jobs = [(nm, tx, c2, c1, rr)
            for nm, tx, c2, c1 in keys
            for rr in range(RNG_START, RNG_START + RUNS)]

    total = len(jobs)
    print(f"Launching {total} simulations on {MAX_WORKERS} workers ...\n")

    # raw[(nMpdus, txopMs, ch1Occ)][ch2Occ] = list of per-run DL throughputs
    raw = {(nm, tx, c1): {c2: [] for c2 in CH2_OCC_LIST}
           for nm, tx, c1 in itertools.product(NMDPUS_LIST, TXOP_LIST, CH1_OCC_LIST)}
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut = {ex.submit(run_one, nm, tx, c2, c1, rr): (nm, tx, c2, c1, rr)
               for nm, tx, c2, c1, rr in jobs}
        for future in as_completed(fut):
            nm, tx, c2, c1, rr = fut[future]
            done += 1
            try:
                dl = future.result()
                raw[(nm, tx, c1)][c2].append(dl)
                save_result(nm, tx, c2, c1, rr, dl)
                print(f"  [{done:>3}/{total}]  n={nm:<4}  txop={int(tx)}ms  "
                      f"ch2={int(c2):>2}%  ch1={int(c1):>2}%  "
                      f"rng={rr}  DL={dl:.1f} Mb/s")
            except Exception as e:
                print(f"  [{done:>3}/{total}]  [ERROR] "
                      f"n={nm} txop={tx}ms ch2={c2}% ch1={c1}% rng{rr}: {e}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    means = {k: [] for k in raw}
    hw    = {k: [] for k in raw}
    for k, c2_dict in raw.items():
        nm, tx, c1 = k
        print(f"\nnMpdus={nm}  txop={tx}ms  ch1={c1}%:")
        for c2 in CH2_OCC_LIST:
            mn, h = ci_hw(c2_dict[c2])
            means[k].append(mn)
            hw[k].append(h)
            print(f"  ch2={int(c2):>2}%: {mn:.2f} ± {h:.2f} Mb/s  (n={len(c2_dict[c2])})")

    # ── Save aggregated results to CSV ─────────────────────────────────────
    csv_file = os.path.join(OUTPUT_DIR, "scenario4_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nMpdus", "txopLimitMs", "ch1OccPct", "ch2OccPct",
                         "mean_dl_mbps", "ci_hw_mbps", "n_runs"])
        for k, c2_dict in raw.items():
            nm, tx, c1 = k
            for c2_idx, c2 in enumerate(CH2_OCC_LIST):
                mn = means[k][c2_idx]
                h  = hw[k][c2_idx]
                writer.writerow([nm, tx, int(c1), int(c2),
                                 f"{mn:.4f}", f"{h:.4f}", len(c2_dict[c2])])
    print(f"Results saved to {csv_file}")

    # ── Plot: line + marker chart (6 lines per subplot) ──────────────────────
    # 6 lines = 3 nMpdus × 2 txop combinations
    # 2 subplots side by side: ch1=0% and ch1=10%
    colors     = {64: "#4C72B0", 512: "#DD8452", 1024: "#55A868"}
    linestyles = {0.0: "-",       3.0: "--"}
    markers    = {64: "o",        512: "s",       1024: "^"}
    x          = np.array(CH2_OCC_LIST)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax_idx, c1 in enumerate(CH1_OCC_LIST):
        ax = axes[ax_idx]
        for nm in NMDPUS_LIST:
            for tx in TXOP_LIST:
                k      = (nm, tx, c1)
                mn_arr = np.array(means[k])
                hw_arr = np.array(hw[k])
                ax.errorbar(
                    x, mn_arr, yerr=hw_arr,
                    label=f"N={nm}, TXOP={int(tx)}ms",
                    color=colors[nm],
                    linestyle=linestyles[tx],
                    marker=markers[nm],
                    markersize=6,
                    linewidth=2,
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5,
                    zorder=3,
                )
        ax.set_xlabel("ch106 occupancy [%]", fontsize=12)
        ax.set_title(f"ch42 occupancy = {int(c1)} %", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(c2)}%" for c2 in CH2_OCC_LIST])
        ax.set_xlim(CH2_OCC_LIST[0] - 5, CH2_OCC_LIST[-1] + 5)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        if ax_idx == 0:
            ax.set_ylabel("Main BSS DL Throughput [Mbit/s]", fontsize=12)
        ax.legend(title="A-MPDU / TXOP", fontsize=9, title_fontsize=9,
                  ncol=2, loc="upper right")

    fig.suptitle(
        f"Scenario 4 – STR DL throughput vs. OBSS channel occupancy\n"
        f"5 GHz ch42+ch106, 80 MHz, EHT MCS 11  |  "
        f"Mean ± {int(CI*100)}% CI  |  {RUNS} runs",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    print(f"\nPlot saved to {PLOT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()

