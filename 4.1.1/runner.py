#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim
import os
import csv
import math
import subprocess
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt

# ---- CONFIG ----
NS3_SCRIPT = "scratch/4.1.1/throughput"   # bez .cc
RUNS = 20
PARALLEL_JOBS = 4
RNG_START = 1101

SIMTIME = 30.0
WARMUP = 1.0
PAYLOAD = 1000
MCS = 11
GI_NS = 3200
DIST_M = 1.0

OUTPUT_DIR = "scratch/4.1.1/baseline_thr_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

PLOT_FILE = os.path.join(PLOT_DIR, "4_1_1.svg")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "baseline_throughput_summary.csv")

# 4 scenariusze (jak w tabeli) => 4 punkty
SCENARIOS = [
    ("1L_80",        dict(nLinks=1, channelWidth=80,  ch0=42,  ch1=106)),
    ("1L_160",       dict(nLinks=1, channelWidth=160, ch0=50,  ch1=114)),
    ("2L_80+80",     dict(nLinks=2, channelWidth=80,  ch0=42,  ch1=106)),
    ("2L_160+160",   dict(nLinks=2, channelWidth=160, ch0=50,  ch1=114)),
]

# 2 limity A-MPDU => 8 wyników
AMPDU_LIST = [64, 1024]


def run_one(run_number, scen_name, scen_params, ampdu):
    out_csv = os.path.join(
        OUTPUT_DIR,
        f"thr_{scen_name}_MCS{MCS}_AMPDU{ampdu}_run{run_number}.csv"
    )

    cmd = [
        "./ns3", "run", "--no-build",
        (
            f"{NS3_SCRIPT} "
            f"--RngRun={run_number} "
            f"--simtime={SIMTIME} "
            f"--warmup={WARMUP} "
            f"--payloadSize={PAYLOAD} "
            f"--nLinks={scen_params['nLinks']} "
            f"--channelWidth={scen_params['channelWidth']} "
            f"--ch0={scen_params['ch0']} "
            f"--ch1={scen_params['ch1']} "
            f"--mcs={MCS} "
            f"--ampduMpdus={ampdu} "
            f"--giNs={GI_NS} "
            f"--apStaDistance={DIST_M} "
            f"--thrCsv={out_csv}"
        )
    ]

    print(f"[RUN {run_number}] {scen_name}_AMPDU{ampdu} start")
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        # pokaż logi żeby było widać dlaczego ns-3 padł
        if p.stdout:
            print(p.stdout)
        if p.stderr:
            print(p.stderr)
        raise RuntimeError(f"ns-3 failed (rc={p.returncode}) for {scen_name}, run={run_number}")

    return out_csv


def read_throughput_csv(path):
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        row = next(r, None)
        if row is None:
            raise ValueError(f"Pusty CSV: {path}")

        key = "throughputMbps"   # <-- u Ciebie tak się nazywa kolumna
        if key not in row:
            raise KeyError(f"CSV nie ma kolumny {key}: {path} (fields={r.fieldnames})")

        return float(row[key])


def ci95(values):
    n = len(values)
    if n < 2:
        return 0.0
    s = np.std(values, ddof=1)
    return 1.96 * s / math.sqrt(n)


def main():
    results = {ampdu: {name: [] for name, _ in SCENARIOS} for ampdu in AMPDU_LIST}

    for ampdu in AMPDU_LIST:
        for scen_name, scen_params in SCENARIOS:
            csv_files = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=PARALLEL_JOBS) as ex:
                futures = [
                    ex.submit(run_one, RNG_START + i, scen_name, scen_params, ampdu)
                    for i in range(RUNS)
                ]
                for fu in concurrent.futures.as_completed(futures):
                    csv_files.append(fu.result())

            for path in sorted(csv_files):
                thr = read_throughput_csv(path)
                results[ampdu][scen_name].append(thr)

            print(f"[OK] {scen_name} AMPDU={ampdu}: n={len(results[ampdu][scen_name])}")

    # ---- SUMMARY CSV ----
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "ampdu_mpdus", "n", "mean_mbps", "ci95_mbps"])
        for ampdu in AMPDU_LIST:
            for scen_name, _ in SCENARIOS:
                vals = results[ampdu][scen_name]
                w.writerow([scen_name, ampdu, len(vals), float(np.mean(vals)), float(ci95(vals))])
    print(f"[OK] Summary CSV: {SUMMARY_CSV}")

    # ---- PLOT (grouped bars) ----
    xlabels = [name for name, _ in SCENARIOS]
    x = np.arange(len(xlabels))
    width = 0.35

    means_64 = [np.mean(results[64][name]) for name in xlabels]
    cis_64   = [ci95(results[64][name]) for name in xlabels]

    means_1024 = [np.mean(results[1024][name]) for name in xlabels]
    cis_1024   = [ci95(results[1024][name]) for name in xlabels]

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, means_64, width, yerr=cis_64, capsize=6,
            label="A-MPDU limit = 64 MPDUs (mean ± 95% CI)")
    plt.bar(x + width/2, means_1024, width, yerr=cis_1024, capsize=6,
            label="A-MPDU limit = 1024 MPDUs (mean ± 95% CI)")

    plt.xticks(x, xlabels)
    plt.ylabel("Throughput [Mbps]")
    plt.title(f"Baseline throughput — MCS {MCS}, GI {GI_NS} ns (mean ± 95% CI)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    plt.show()
    print(f"[OK] Plot saved: {PLOT_FILE}")


if __name__ == "__main__":
    main()
