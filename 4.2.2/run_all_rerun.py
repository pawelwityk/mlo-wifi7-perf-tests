#!/usr/bin/env python3
"""
run_all_rerun.py  —  Latency Scenario #2 (4.2.2)
Sweep: numLinks × offeredLoad × RngRun
Persistence: per-run CSV result files (resume after crash / add more runs).
Re-run: failed jobs retried sequentially after the parallel pass.
Plot: latency and throughput vs. offered load, one line per numLinks value.
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
NS3_BINARY   = "build/scratch/4.2.2/ns3.45-wifi-mlo-latency-scenario2-optimized"
OUTPUT_DIR   = "thr_runs_4_2_2_precise"
PLOT_DIR     = "plots"
PLOT_FILE    = os.path.join(PLOT_DIR, "4_2_2.pdf")
PLOT_FILE_PCTL_THR = os.path.join(PLOT_DIR, "4_2_2_p50_p99_vs_throughput.pdf")
PLOT_FILE_PCTL_LOAD = os.path.join(PLOT_DIR, "4_2_2_p50_p99_vs_load.pdf")

NUM_LINKS      = [1, 2, 4]
OFFERED_LOADS  = list(range(100, 3100, 100))
# Extra fine-grained loads near each numLinks's saturation knee, so the
# p50/p99 curves rise smoothly to the 8 ms ceiling instead of jumping
# over the transition zone (matching praca_mgr Fig. 4.8 style).
FINE_LOADS     = {
    # Fine-grained sampling right AT the saturation knee (where p99 latency
    # peaks before queue truncation kicks in).  NSS=2 actual saturation
    # measured from smoke tests:
    #   L=1: ~813 Mb/s,  L=2: ~1615 Mb/s,  L=4: ~2559 Mb/s
    1: list(range(750, 880, 10)),    # SLO knee  ~813 Mb/s
    2: list(range(1500, 1750, 10)),  # STR-2 knee ~1615 Mb/s
    # STR-4 has so much headroom (4 x ~620 Mb/s useful = 2480 Mb/s) that the
    # standard 100..3000 sweep barely scratches saturation. Push further with
    # extra loads above 3000 Mb/s so the p99 peak becomes visible.
    4: (list(range(2450, 2700, 10)) +
        list(range(3000, 4500, 100)) +
        list(range(4500, 6000, 250))),
}
RUNS           = 20
SIM_TIME       = 20.0
PAYLOAD        = 1500
NMDPUS         = 1024

MAX_WORKERS    = 11
RERUN_MISSING  = True
STARTUP_GUARD  = 2.0
MAX_MAC_QUEUE_PACKETS = 4096
MAX_PLOT_P99_MS = 6.0

os.makedirs(PLOT_DIR, exist_ok=True)

RE_LAT = re.compile(r"Mean DL Latency:\s*([0-9.]+)\s*ms")
RE_P50 = re.compile(r"DL Latency p50:\s*([0-9.]+)\s*ms")
RE_P99 = re.compile(r"DL Latency p99:\s*([0-9.]+)\s*ms")
RE_THR = re.compile(r"Mean DL Throughput:\s*([0-9.]+)\s*Mb/s")


# ---------------------------------------------------------------------------
#  Persistence helpers
# ---------------------------------------------------------------------------

def _result_path(nl: int, load: int, rng: int) -> str:
    fname = f"L{nl}_Load{load}_rng{rng}_result.csv"
    return os.path.join(OUTPUT_DIR, fname)


def save_result(
    nl: int,
    load: int,
    rng: int,
    lat_mean: float,
    lat_p50: float,
    lat_p99: float,
    thr: float
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(_result_path(nl, load, rng), "w") as f:
        f.write(f"{lat_mean},{lat_p50},{lat_p99},{thr}\n")


def load_existing_results(raw: dict) -> set:
    done = set()

    if not os.path.isdir(OUTPUT_DIR):
        return done

    for fname in os.listdir(OUTPUT_DIR):
        if not fname.endswith("_result.csv"):
            continue

        m = re.match(r"^L(\d+)_Load(\d+)_rng(\d+)_result\.csv$", fname)
        if not m:
            continue

        nl = int(m.group(1))
        load = int(m.group(2))
        rng = int(m.group(3))
        path = os.path.join(OUTPUT_DIR, fname)

        try:
            with open(path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            if not lines:
                continue

            if len(lines) >= 2 and lines[0].startswith("numLinks,"):
                parts = [p.strip() for p in lines[1].split(",")]
                if len(parts) < 5:
                    raise ValueError(f"legacy row malformed: {lines[1]!r}")

                lat_mean = float(parts[3])
                lat_p50 = lat_mean
                lat_p99 = lat_mean
                thr = float(parts[4])

            else:
                parts = [p.strip() for p in lines[0].split(",")]

                if len(parts) == 4:
                    lat_mean = float(parts[0])
                    lat_p50 = float(parts[1])
                    lat_p99 = float(parts[2])
                    thr = float(parts[3])
                elif len(parts) >= 2:
                    lat_mean = float(parts[0])
                    lat_p50 = lat_mean
                    lat_p99 = lat_mean
                    thr = float(parts[1])
                else:
                    raise ValueError(f"plain row malformed: {lines[0]!r}")

            key = (nl, load)
            raw.setdefault(key, {
                "latMean": [],
                "latP50": [],
                "latP99": [],
                "thr": [],
            })

            raw[key]["latMean"].append(lat_mean)
            raw[key]["latP50"].append(lat_p50)
            raw[key]["latP99"].append(lat_p99)
            raw[key]["thr"].append(thr)

            done.add((nl, load, rng))

        except Exception as exc:
            print(f"[warn] Could not read {fname}: {exc}", flush=True)

    return done


# ---------------------------------------------------------------------------
#  Run one simulation
# ---------------------------------------------------------------------------

def run_sim(nl: int, load: int, rng: int) -> tuple[float, float, float, float] | None:
    cmd = [
        NS3_BINARY,
        f"--simTime={SIM_TIME}",
        f"--payloadSize={PAYLOAD}",
        f"--nMpdus={NMDPUS}",
        f"--numLinks={nl}",
        f"--offeredLoad={load}",
        f"--startupGuard={STARTUP_GUARD}",
        f"--maxMacQueuePackets={MAX_MAC_QUEUE_PACKETS}",
        f"--RngRun={rng}",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"[timeout] L={nl} load={load} rng={rng}", flush=True)
        return None

    if result.returncode != 0:
        print(f"[error] L={nl} load={load} rng={rng}", flush=True)
        print(result.stderr[-2000:], flush=True)
        return None

    output = result.stdout + result.stderr

    m_lat = RE_LAT.search(output)
    m_p50 = RE_P50.search(output)
    m_p99 = RE_P99.search(output)
    m_thr = RE_THR.search(output)

    if not m_lat or not m_thr:
        print(f"[parse-fail] L={nl} load={load} rng={rng}", flush=True)
        return None

    lat_mean = float(m_lat.group(1))
    lat_p50 = float(m_p50.group(1)) if m_p50 else lat_mean
    lat_p99 = float(m_p99.group(1)) if m_p99 else lat_mean
    thr = float(m_thr.group(1))

    return lat_mean, lat_p50, lat_p99, thr


# ---------------------------------------------------------------------------
#  Plot: mean latency
# ---------------------------------------------------------------------------

def plot_results(raw: dict) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2))
    fig.suptitle("Latency Scenario #2 — Mean DL Latency vs. Achieved Throughput")

    colors = {
        1: "tab:blue",
        2: "tab:orange",
        4: "tab:green",
    }

    labels = {
        1: "SLO (1 link)",
        2: "STR (2 links)",
        4: "STR (4 links)",
    }

    for nl in NUM_LINKS:
        rows = []

        for load in sorted(set(OFFERED_LOADS + FINE_LOADS.get(nl, []))):
            key = (nl, load)

            if key not in raw or len(raw[key]["latMean"]) == 0:
                continue

            from _robust import ci_hw_robust

            lat_m, ci_lat = ci_hw_robust(raw[key]["latMean"])
            thr_m, ci_thr = ci_hw_robust(raw[key]["thr"])

            rows.append((thr_m, lat_m, ci_lat))

        if not rows:
            continue

        rows.sort(key=lambda r: r[0])

        x = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        yerr = np.array([r[2] for r in rows])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=labels[nl],
            color=colors[nl],
            marker="o",
            markersize=4,
            capsize=3,
        )

    ax.set_xlabel("Achieved Throughput [Mbit/s]")
    ax.set_ylabel("Channel access delay [ms]")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 6)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="pdf", dpi=300)
    print(f"\n[plot] saved → {PLOT_FILE}", flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Percentile helpers
# ---------------------------------------------------------------------------

def _percentile_series(raw: dict, nl: int):
    from _robust import remove_outliers_robust

    rows = []

    for load in sorted(set(OFFERED_LOADS + FINE_LOADS.get(nl, []))):
        key = (nl, load)

        if key not in raw or len(raw[key]["latP50"]) == 0:
            continue

        p50s = np.array(raw[key]["latP50"], dtype=float)
        p99s = np.array(raw[key]["latP99"], dtype=float)
        thrs = np.array(raw[key]["thr"], dtype=float)

        y50 = float(np.mean(remove_outliers_robust(p50s)))
        y99 = float(np.mean(remove_outliers_robust(p99s)))
        y99 = max(y99, y50)

        xthr = float(np.mean(remove_outliers_robust(thrs)))

        if y50 <= 0:
            continue

        rows.append((float(load), xthr, y50, y99))

    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([])

    rows.sort(key=lambda r: r[0])

    load = np.array([r[0] for r in rows])
    thr = np.array([r[1] for r in rows])
    y50 = np.array([r[2] for r in rows])
    y99 = np.array([r[3] for r in rows])

    return thr, y50, y99, load


# ---------------------------------------------------------------------------
#  Plot: p50-p99 bands
# ---------------------------------------------------------------------------

def plot_percentile_bands(raw: dict) -> None:
    colors = {
        1: "tab:blue",
        2: "tab:orange",
        4: "tab:green",
    }

    labels = {
        1: "SL (1 link)",
        2: "STR (2 links)",
        4: "STR (4 links)",
    }

    for plot_file, use_load, xlabel in [
        (PLOT_FILE_PCTL_THR, False, "Achieved Throughput [Mbit/s]"),
        (PLOT_FILE_PCTL_LOAD, True, "Offered Load [Mbit/s]"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2))

        suffix = "offered load" if use_load else "achieved throughput"
        ax.set_title(f"Scenario 4.2.2: latency p50–p99 vs {suffix}")

        for nl in NUM_LINKS:
            x_thr, y50, y99, x_load = _percentile_series(raw, nl)

            if x_thr.size == 0:
                continue

            order = np.argsort(x_load)

            xvals = x_load if use_load else x_thr
            xv = xvals[order]
            y50s = y50[order]
            y99s = y99[order]

            xv = np.concatenate(([0.0], xv))
            y50s = np.concatenate(([0.0], y50s))
            y99s = np.concatenate(([0.0], y99s))

            ymax = MAX_PLOT_P99_MS
            stable = y99s < ymax
            n_stab = int(stable.sum())

            if n_stab < 2:
                continue

            xs = xv[:n_stab]
            ys50 = y50s[:n_stab]
            ys99 = y99s[:n_stab]

            # -------------------------------------------------------------------
            # THE FIX: BYPASS THE BOOBY-TRAPPED fill_between
            # -------------------------------------------------------------------
            # `thesis_style` heavily monkey-patches `ax.fill_between` to squeeze 
            # the visual width of error bands to a max of 10%. By manually 
            # calculating the polygon coordinates and using `ax.fill()` instead, 
            # we completely evade the monkey-patch and draw exactly what you expect.
            
            x_poly = np.concatenate([xs, xs[::-1]])
            y_poly = np.concatenate([ys50, ys99[::-1]])
            
            ax.fill(
                x_poly, 
                y_poly, 
                color=colors[nl], 
                alpha=0.25, 
                linewidth=0,
                zorder=1
            )
            # -------------------------------------------------------------------

            ax.plot(
                xs,
                ys50,
                color=colors[nl],
                linestyle="--",
                linewidth=1.8,
                label=f"{labels[nl]} p50",
                zorder=2
            )

            ax.plot(
                xs,
                ys99,
                color=colors[nl],
                linestyle="-",
                linewidth=2.2,
                label=f"{labels[nl]} p99",
                zorder=2
            )

        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Channel access delay [ms]")
        ax.set_xlim(left=0)
        ax.set_ylim(0, MAX_PLOT_P99_MS)
        
        ax.legend(ncol=2, fontsize=9)

        plt.tight_layout()
        plt.savefig(plot_file, format="pdf", bbox_inches="tight")
        print(f"[plot] saved → {plot_file}", flush=True)
        plt.close(fig)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw: dict = {}
    already_done = load_existing_results(raw)

    print(f"[resume] {len(already_done)} run(s) already completed.", flush=True)

    all_jobs = [
        (nl, load, rng)
        for nl in NUM_LINKS
        for load in sorted(set(OFFERED_LOADS + FINE_LOADS.get(nl, [])))
        for rng in range(1, RUNS + 1)
    ]

    total = len(all_jobs)
    pending_jobs = [j for j in all_jobs if j not in already_done]

    print(f"[plan] {len(pending_jobs)} / {total} runs pending.", flush=True)

    failed: list[tuple] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_job = {
            ex.submit(run_sim, nl, load, rng): (nl, load, rng)
            for nl, load, rng in pending_jobs
        }

        for future in as_completed(future_to_job):
            nl, load, rng = future_to_job[future]
            res = future.result()
            completed += 1

            if res is None:
                failed.append((nl, load, rng))
                print(
                    f"[FAIL  {completed}/{len(pending_jobs)}] "
                    f"L={nl} load={load} rng={rng}",
                    flush=True,
                )

            else:
                lat_mean, lat_p50, lat_p99, thr = res
                key = (nl, load)

                raw.setdefault(key, {
                    "latMean": [],
                    "latP50": [],
                    "latP99": [],
                    "thr": [],
                })

                raw[key]["latMean"].append(lat_mean)
                raw[key]["latP50"].append(lat_p50)
                raw[key]["latP99"].append(lat_p99)
                raw[key]["thr"].append(thr)

                save_result(nl, load, rng, lat_mean, lat_p50, lat_p99, thr)

                print(
                    f"[ok    {completed}/{len(pending_jobs)}] "
                    f"L={nl} load={load} rng={rng}  "
                    f"mean={lat_mean:.3f} ms "
                    f"p50={lat_p50:.3f} ms "
                    f"p99={lat_p99:.3f} ms "
                    f"thr={thr:.2f} Mb/s",
                    flush=True,
                )

    if RERUN_MISSING and failed:
        print(
            f"\n[rerun] {len(failed)} failed runs — retrying sequentially …\n",
            flush=True,
        )

        still_failed: list[tuple] = []

        for idx, (nl, load, rng) in enumerate(failed, 1):
            res = run_sim(nl, load, rng)

            if res is None:
                still_failed.append((nl, load, rng))
                print(
                    f"[FAIL-rerun {idx}/{len(failed)}] "
                    f"L={nl} load={load} rng={rng}",
                    flush=True,
                )

            else:
                lat_mean, lat_p50, lat_p99, thr = res
                key = (nl, load)

                raw.setdefault(key, {
                    "latMean": [],
                    "latP50": [],
                    "latP99": [],
                    "thr": [],
                })

                raw[key]["latMean"].append(lat_mean)
                raw[key]["latP50"].append(lat_p50)
                raw[key]["latP99"].append(lat_p99)
                raw[key]["thr"].append(thr)

                save_result(nl, load, rng, lat_mean, lat_p50, lat_p99, thr)

                print(
                    f"[ok-rerun {idx}/{len(failed)}] "
                    f"L={nl} load={load} rng={rng}  "
                    f"mean={lat_mean:.3f} ms "
                    f"p50={lat_p50:.3f} ms "
                    f"p99={lat_p99:.3f} ms "
                    f"thr={thr:.2f} Mb/s",
                    flush=True,
                )

        if still_failed:
            print(
                f"\n[warn] {len(still_failed)} run(s) still failed after rerun:",
                flush=True,
            )

            for job in still_failed:
                print(
                    f"  L={job[0]} load={job[1]} rng={job[2]}",
                    flush=True,
                )

    plot_results(raw)
    plot_percentile_bands(raw)


if __name__ == "__main__":
    main()