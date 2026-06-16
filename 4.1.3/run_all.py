"""
Runner + plotter for scratch/4.1.3/wifi-mlo-throughput-scenario1-yans
Two modes: STR and EMLSR (BSS2 only; BSS0/BSS1 always SLO).
10 independent runs each, per-BSS throughput, 95 % CI bars.
Reduce RUNS (e.g. to 5) for faster sweeps at the cost of wider CI bars.
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
NS3_PROGRAM = "scratch/4.1.3/wifi-mlo-throughput-scenario1-yans"
SIMTIME     = 10.0
PAYLOAD     = 1500
NMDPUS      = 1024
RUNS        = 10
RNG_START   = 1001
CI          = 0.95
MAX_WORKERS = os.cpu_count() or 8
OUTPUT_DIR  = "thr_runs_4_1_3"
PLOT_DIR    = "plots"
PLOT_FILE   = os.path.join(PLOT_DIR, "4_1_3.svg")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

MODES = ["STR", "EMLSR"]   # only affects BSS2
BSS_LABELS = ["BSS0\n(SLO ch42)", "BSS1\n(SLO ch106)", "BSS2\n(MLO ch42+106)"]

RE_BSS = [
    re.compile(r"MacRx BSS0[^:]*:\s*([0-9.]+)\s*Mb/s"),
    re.compile(r"MacRx BSS1[^:]*:\s*([0-9.]+)\s*Mb/s"),
    re.compile(r"MacRx BSS2[^:]*:\s*([0-9.]+)\s*Mb/s"),
]


# ---------------------------------------------------------------------------

def run_one(mode, rng_run):
    """Return (bss0, bss1, bss2) Mb/s or raise."""
    args = (
        f"{NS3_PROGRAM} "
        f"--RngRun={rng_run} "
        f"--simTime={SIMTIME} "
        f"--payloadSize={PAYLOAD} "
        f"--nMpdus={NMDPUS} "
        f"--mloMode={mode}"
    )
    cmd = ["./ns3", "run", args]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log = os.path.join(OUTPUT_DIR, f"{mode}_rng{rng_run}_err.txt")
        with open(log, "w") as f:
            f.write("CMD:\n" + " ".join(cmd)
                    + "\n\nSTDOUT:\n" + result.stdout
                    + "\n\nSTDERR:\n" + result.stderr)
        raise RuntimeError(f"exit={result.returncode}, log={log}")

    vals = [None, None, None]
    for line in result.stdout.splitlines():
        for i, rx in enumerate(RE_BSS):
            m = rx.search(line)
            if m:
                vals[i] = float(m.group(1))

    if any(v is None for v in vals):
        log = os.path.join(OUTPUT_DIR, f"{mode}_rng{rng_run}_stdout.txt")
        with open(log, "w") as f:
            f.write(result.stdout)
        raise RuntimeError(f"Missing MacRx line(s): {vals}, log={log}")

    return tuple(vals)


def ci_hw(data, confidence=CI):
    n = len(data)
    if n < 2:
        return float(np.mean(data)), 0.0
    mean = float(np.mean(data))
    h    = float(stats.sem(data) * stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return mean, h


# ---------------------------------------------------------------------------

def main():
    # Build job list: (mode, rng_run)
    jobs = [(mode, rr)
            for mode in MODES
            for rr in range(RNG_START, RNG_START + RUNS)]

    total = len(jobs)
    print(f"Launching {total} simulations on {MAX_WORKERS} workers ...\n")

    # raw[mode][bss_idx] = list of floats
    raw = {m: [[] for _ in range(3)] for m in MODES}
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut = {ex.submit(run_one, mode, rr): (mode, rr) for mode, rr in jobs}
        for future in as_completed(fut):
            mode, rr = fut[future]
            done += 1
            try:
                b0, b1, b2 = future.result()
                raw[mode][0].append(b0)
                raw[mode][1].append(b1)
                raw[mode][2].append(b2)
                print(f"  [{done:>3}/{total}]  {mode:<5}  RngRun={rr}"
                      f"  BSS0={b0:.1f}  BSS1={b1:.1f}  BSS2={b2:.1f}  Mb/s")
            except Exception as e:
                print(f"  [{done:>3}/{total}]  [ERROR]  {mode}  RngRun={rr}: {e}")

    # Aggregate
    # means[mode][bss], hw[mode][bss]
    means = {m: [] for m in MODES}
    hw    = {m: [] for m in MODES}
    for m in MODES:
        for b in range(3):
            mn, h = ci_hw(raw[m][b])
            means[m].append(mn)
            hw[m].append(h)
        print(f"\n{m}:")
        for b in range(3):
            print(f"  BSS{b}: {means[m][b]:.2f} ± {hw[m][b]:.2f} Mb/s  (n={len(raw[m][b])})")

    # ------------------------------------------------------------------ plot
    n_modes = len(MODES)  # 2 main columns: STR and EMLSR
    n_bss   = 3
    x       = np.arange(n_modes)
    width   = 0.25
    # Colors per BSS instead of per mode
    colors  = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}  # BSS0, BSS1, BSS2
    bss_labels_short = ["BSS0 (SLO ch42)", "BSS1 (SLO ch106)", "BSS2 (MLO ch42+106)"]
    
    # Offsets for 3 bars within each mode group
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(11, 6))

    # For each BSS, plot bars across the modes
    for bss_idx in range(n_bss):
        mode_values = []
        mode_errs = []
        for m in MODES:
            mode_values.append(means[m][bss_idx])
            mode_errs.append(hw[m][bss_idx])
        
        bars = ax.bar(
            x + offsets[bss_idx],
            mode_values,
            width,
            yerr=mode_errs,
            capsize=6,
            label=bss_labels_short[bss_idx],
            color=colors[bss_idx],
            edgecolor="black",
            linewidth=0.8,
            error_kw=dict(elinewidth=1.8, ecolor="black", capthick=1.8),
            zorder=3,
        )
        for bar, mn, h in zip(bars, mode_values, mode_errs):
            top = float(mn + h)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                top + ax.get_ylim()[1] * 0.01,   # small offset; updated after autoscale
                f"{mn:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MODES, fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput [Mbit/s]", fontsize=12)
    ax.set_xlabel("BSS2 Mode", fontsize=12)

    all_tops = []
    for m in MODES:
        all_tops += [means[m][b] + hw[m][b] for b in range(3)]
    y_top = max(all_tops) if all_tops else 1.0
    ax.set_ylim(0, y_top * 1.25)

    # Reposition value labels now that ylim is known
    for ax_child in ax.texts:
        ax_child.set_y(ax_child.get_position()[1])   # keep as-is; already absolute

    ax.set_title(
        f"Coexistence scenario – per-BSS MacRx throughput\n"
        f"BSS2 mode: STR vs EMLSR  |  "
        f"Mean ± {int(CI*100)}% CI  |  {RUNS} runs",
        fontsize=12,
    )
    ax.legend(title="BSS Configuration", fontsize=10, title_fontsize=10, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    plt.show()
    print(f"\n[OK] Plot saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()
