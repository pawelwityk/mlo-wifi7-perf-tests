# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import re

# ===== KONFIG =====

# ŚCIEŻKA DO BINARKI (po: ./ns3 build scratch/4.1.2/wifi-mlo-throughput-scenario1.cc)
RUNS = 10
SIMTIME = 30.0
PAYLOAD = 1500
N_MPDUS = 1024
RNG_START = 1001

OUTPUT_DIR = "thr_runs_all"
SUMMARY_CSV = "throughput_summary_all.csv"
PLOT_DIR = "plots"
PLOT_FILE = os.path.join(PLOT_DIR, "4_1_2.svg")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# (etykieta, mloMode, numLinks)
CONFIGS = [
    ("SL 1 link",      "SLO",   1),
    ("EMLSR 2 links",  "EMLSR", 2),
    ("STR 2 links",    "STR",   2),
    ("EMLSR 3 links",  "EMLSR", 3),
    ("STR 3 links",    "STR",   3),
]

# ===== PARSERY (dopasowane do Twoich printfów) =====
RE_L3 = re.compile(r"Average L3 throughput \(FlowMonitor\):\s*([0-9.]+)\s*Mb/s")
RE_MAC = re.compile(r"MAC-level throughput .*:\s*([0-9.]+)\s*Mb/s")
RE_PHY_EFF = re.compile(r"PHY-level effective throughput over sim time:\s*([0-9.]+)\s*Mb/s")
RE_PHY_RATE = re.compile(r"PHY-level data rate \(bits/airtime\):\s*([0-9.]+)\s*Mb/s")
RE_AIRTIME = re.compile(r"PHY airtime utilization:\s*([0-9.]+)\s*%")

def run_one_simulation(cfg_label, mlo_mode, num_links, rng_run):
    cmd = [
    "./ns3", "run",
    f"scratch/4.1.2/wifi-mlo-throughput-scenario1-traces "
    f"--RngRun={rng_run} --simTime={SIMTIME} --payloadSize={PAYLOAD} "
    f"--nMpdus={N_MPDUS} --numLinks={num_links} --mloMode={mlo_mode}"
    ]
    print(f"[{cfg_label}] RUN {rng_run}  Starting...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        safe_label = cfg_label.replace(" ", "_")
        log_path = os.path.join(OUTPUT_DIR, f"{safe_label}_rng{rng_run}_stderr.txt")
        with open(log_path, "w") as f:
            f.write("STDOUT:\n" + stdout + "\n\nSTDERR:\n" + stderr)
        raise RuntimeError(
            f"[{cfg_label}] RUN {rng_run} exit={result.returncode}, log={log_path}"
        )

    l3 = mac = phy_eff = phy_rate = airtime_pct = None

    for line in stdout.splitlines():
        m = RE_L3.search(line)
        if m:
            l3 = float(m.group(1))
            continue

        m = RE_MAC.search(line)
        if m:
            mac = float(m.group(1))
            continue

        m = RE_PHY_EFF.search(line)
        if m:
            phy_eff = float(m.group(1))
            continue

        m = RE_PHY_RATE.search(line)
        if m:
            phy_rate = float(m.group(1))  # <-- TO traktujemy jako L1 throughput
            continue

        m = RE_AIRTIME.search(line)
        if m:
            airtime_pct = float(m.group(1))
            continue

    # Wymagamy L1 = PHY data rate (bits/airtime)
    if phy_rate is None:
        safe_label = cfg_label.replace(" ", "_")
        log_path = os.path.join(OUTPUT_DIR, f"{safe_label}_rng{rng_run}_stdout.txt")
        with open(log_path, "w") as f:
            f.write(stdout)
        raise RuntimeError(
            f"[{cfg_label}] RUN {rng_run} brak PHY-level data rate (bits/airtime), log={log_path}"
        )

    print(f"[{cfg_label}] RUN {rng_run}  L1(PHY bits/airtime) = {phy_rate:.3f} Mb/s")
    return {
        "L3_Mbps": l3,
        "MAC_Mbps": mac,
        "L1_PHY_bits_airtime_Mbps": phy_rate,
        "PHY_eff_over_sim_Mbps": phy_eff,
        "PHY_airtime_pct": airtime_pct,
    }

def main():
    all_rows = []
    cfg_labels = []
    cfg_means = []

    current_rng = RNG_START

    for label, mlo_mode, num_links in CONFIGS:
        print("\n====================================")
        print(f"Konfiguracja: {label}")
        print(f"  mloMode  = {mlo_mode}")
        print(f"  numLinks = {num_links}")
        print("====================================\n")

        rng_values = list(range(current_rng, current_rng + RUNS))
        current_rng += RUNS

        results = []  # (rr, dict)

        for rr in rng_values:
            try:
                res = run_one_simulation(label, mlo_mode, num_links, rr)
                results.append((rr, res))
            except Exception as e:
                print(f"[ERROR] {label} / RngRun={rr}: {e}")

        if not results:
            print(f"[WARN] Brak wyników dla {label}, pomijam.")
            continue

        results.sort(key=lambda x: x[0])

        # === ŚREDNIA po L1 = PHY bits/airtime ===
        l1_vals = np.array([r["L1_PHY_bits_airtime_Mbps"] for (_, r) in results], dtype=float)
        mean_l1 = float(np.mean(l1_vals))

        cfg_labels.append(label)
        cfg_means.append(mean_l1)

        print(f"\n>>> {label}: mean L1(PHY bits/airtime) = {mean_l1:.3f} Mb/s "
              f"(runs={len(l1_vals)})\n")

        for idx, (rr, r) in enumerate(results, start=1):
            all_rows.append({
                "config_label": label,
                "mloMode": mlo_mode,
                "numLinks": num_links,
                "run_idx": idx,
                "RngRun": rr,
                "L1_PHY_bits_airtime_Mbps": r["L1_PHY_bits_airtime_Mbps"],
                "L3_Mbps": r["L3_Mbps"],
                "MAC_Mbps": r["MAC_Mbps"],
                "PHY_eff_over_sim_Mbps": r["PHY_eff_over_sim_Mbps"],
                "PHY_airtime_pct": r["PHY_airtime_pct"],
            })

    if not cfg_means:
        print("[ERROR] Nie udało się zebrać żadnych wyników.")
        return

    # CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        fieldnames = [
            "config_label", "mloMode", "numLinks",
            "run_idx", "RngRun",
            "L1_PHY_bits_airtime_Mbps",
            "L3_Mbps",
            "MAC_Mbps",
            "PHY_eff_over_sim_Mbps",
            "PHY_airtime_pct",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print(f"[OK] Zapisano pełne wyniki do {SUMMARY_CSV}")

    # Wykres (L1)
    cfg_means = np.array(cfg_means, dtype=float)
    baseline = cfg_means[0]
    rel = (cfg_means - baseline) / baseline * 100.0

    x = np.arange(len(cfg_labels))
    plt.figure(figsize=(7, 6))
    bars = plt.bar(x, cfg_means)

    plt.xticks(x, cfg_labels)
    plt.ylabel("L1 Throughput (PHY bits/airtime) [Mbps]")
    plt.ylim(0, cfg_means.max() * 1.25)
    plt.title("L1 (PHY bits/airtime) vs. MLO mode / number of links")

    for i, (b, r) in enumerate(zip(bars, rel)):
        h = b.get_height()
        if i == 0:
            text = ""
        else:
            sign = "+" if r >= 0 else ""
            text = f"{sign}{r:.1f}%"
        if text:
            plt.text(
                b.get_x() + b.get_width() / 2.0,
                h + cfg_means.max() * 0.03,
                text,
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    plt.show()
    print(f"[OK] Zapisano wykres: {PLOT_FILE}")

if __name__ == "__main__":
    main()
