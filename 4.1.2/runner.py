# >>> thesis-style shim (auto-injected)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401  installs unified plot style
# <<< thesis-style shim
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import os
import csv
import re

# --- CONFIG ---
# Nazwa programu ns-3:
# jeśli plik C++ to scratch/wifi-mlo-throughput-scenario1.cc,
# to tutaj daj "wifi-mlo-throughput-scenario1"
NS3_SCRIPT = "scratch/4.1.2/wifi-mlo-throughput-scenario1.cc"

RUNS = 20              # liczba powtórzeń
SIMTIME = 30.0         # czas symulacji [s]
PAYLOAD = 1500         # rozmiar pakietu [B] (musi zgadzać się z C++)
N_MPDUS = 1024         # limit MPDU (nMpdus w C++)
NUM_LINKS = 3          # 1, 2 lub 3
MLO_MODE = "EMLSR"     # "SLO", "STR" lub "EMLSR"

PARALLEL_JOBS = 4      # liczba równoległych procesów
RNG_START = 1001       # początkowy RNG run

OUTPUT_DIR = "thr_runs"
PLOT_DIR = "plots"
PLOT_FILE = os.path.join(PLOT_DIR, "4_1_2.svg")
THR_CSV = f"throughput_{MLO_MODE}_{NUM_LINKS}links.csv"

os.makedirs(PLOT_DIR, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Uruchomienie jednej symulacji
# --------------------------
def run_one_simulation(run_number):
    """
    Uruchamia jedną symulację NS-3 z zadanym RNG run
    i zwraca zmierzoną wartość throughputu [Mb/s].
    """
    cmd = [
        "./ns3", "run",
        (
            f"{NS3_SCRIPT} "
            f"--RngRun={run_number} "
            f"--simTime={SIMTIME} "
            f"--payloadSize={PAYLOAD} "
            f"--nMpdus={N_MPDUS} "
            f"--numLinks={NUM_LINKS} "
            f"--mloMode={MLO_MODE}"
        )
    ]

    print(f"[RUN {run_number}] Starting...")
    # przechwytujemy stdout, żeby znaleźć "Average throughput: X Mb/s"
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = result.stdout
    print(f"[RUN {run_number}] Done")

    # Szukamy linijki z throughputem
    # Format: "Average throughput: <liczba> Mb/s"
    thr = None
    for line in stdout.splitlines():
        m = re.search(r"Average throughput:\s*([0-9.]+)\s*Mb/s", line)
        if m:
            thr = float(m.group(1))
            break

    if thr is None:
        # Jak coś poszło nie tak, zapisujemy stdout do pliku, żeby podejrzeć
        log_path = os.path.join(OUTPUT_DIR, f"run{run_number}_stdout.txt")
        with open(log_path, "w") as f:
            f.write(stdout)
        raise RuntimeError(f"[RUN {run_number}] Nie znaleziono throughputu w output. Zapisano {log_path}")

    print(f"[RUN {run_number}] Throughput = {thr:.3f} Mb/s")
    return thr

# --------------------------
# Główna funkcja
# --------------------------
def main():
    print(
        f"Running {RUNS} simulations (RngRun {RNG_START}-{RNG_START + RUNS - 1}) "
        f"with {PARALLEL_JOBS} parallel jobs...\n"
        f"Mode={MLO_MODE}, numLinks={NUM_LINKS}"
    )

    throughputs = []  # lista (rng_run, thr)

    # --- URUCHOM WSZYSTKIE SYMULACJE ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
        futures = {
            executor.submit(run_one_simulation, RNG_START + i): (RNG_START + i)
            for i in range(RUNS)
        }
        for future in concurrent.futures.as_completed(futures):
            run_number = futures[future]
            try:
                thr = future.result()
                throughputs.append((run_number, thr))
            except Exception as e:
                print(f"[ERROR] Run {run_number}: {e}")

    if not throughputs:
        print("[ERROR] Brak wyników throughputu.")
        return

    # Sortujemy po RngRun
    throughputs.sort(key=lambda x: x[0])
    rng_runs = np.array([x[0] for x in throughputs], dtype=int)
    thr_vals = np.array([x[1] for x in throughputs], dtype=float)

    # --- Statystyki ---
    mean_thr = np.mean(thr_vals)
    std_thr = np.std(thr_vals, ddof=1) if len(thr_vals) > 1 else 0.0
    print("\n=== Statystyki throughputu ===")
    print(f"Liczba przebiegów : {len(thr_vals)}")
    print(f"Średni throughput  : {mean_thr:.3f} Mb/s")
    print(f"Odchylenie std     : {std_thr:.3f} Mb/s")

    # --- Zapis do CSV ---
    with open(THR_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_idx", "RngRun", "throughput_Mbps"])
        for idx, (rr, thr) in enumerate(throughputs, start=1):
            w.writerow([idx, rr, f"{thr:.6f}"])
    print(f"[OK] Zapisano wyniki throughputu: {THR_CSV}")

    # --- Wykres: throughput vs numer przebiegu + linia średniej ---
    plt.figure(figsize=(7, 5))
    x = np.arange(1, len(thr_vals) + 1)

    plt.plot(x, thr_vals, marker="o", linestyle="-", label="Per-run throughput")
    plt.axhline(mean_thr, linestyle="--", label=f"Mean = {mean_thr:.2f} Mb/s")

    plt.xlabel("Run index")
    plt.ylabel("Average throughput [Mb/s]")
    plt.title(f"Average throughput per run — {MLO_MODE}, {NUM_LINKS} link(s)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    plt.show()
    print(f"[OK] Zapisano wykres: {PLOT_FILE}")

if __name__ == "__main__":
    main()
