#!/usr/bin/env python3
"""
Reduced 4.3 sweep — only Cases A and B (mapped to EMLSR/STR for thesis Figs
4.14–4.16) with a smaller STA count and shorter simulation time so the
sweep finishes in a tractable wall-clock budget.

Override of scratch/4.3/run_all_rerun.py constants.
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scratch", "4.3"))

import run_all_rerun as r

# Trim sweep
r.CASES        = ["A", "B"]
r.LEGACY_PCTS  = list(range(10, 100, 10))
r.RUNS         = int(os.environ.get("RUNS_4_3", "20"))
r.SIM_TIME     = float(os.environ.get("SIMTIME_4_3", "5.0"))
r.TOTAL_STAS   = int(os.environ.get("STAS_4_3", "20"))
r.MAX_WORKERS  = int(os.environ.get("MAX_WORKERS", "6"))
r.NS3_BINARY   = os.path.join(ROOT, "build", "scratch", "4.3",
                              "ns3.45-wifi-mlo-legacy-coexistence-optimized")

print(f"[reduced] CASES={r.CASES}  LEGACY_PCTS={r.LEGACY_PCTS}  "
      f"RUNS={r.RUNS}  SIM_TIME={r.SIM_TIME}  TOTAL_STAS={r.TOTAL_STAS}", flush=True)


def main():
    if hasattr(r, "main"):
        return r.main()
    # Fall back: call the sweep loop directly using its module functions.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from itertools import product
    jobs = list(product(r.CASES, r.LEGACY_PCTS, range(1, r.RUNS + 1)))
    print(f"[run] {len(jobs)} jobs", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=r.MAX_WORKERS) as pool:
        futs = [pool.submit(r.run_job, c, p, n) for c, p, n in jobs]
        for fut in as_completed(futs):
            done += 1
            if done % 10 == 0:
                print(f"  ... {done}/{len(jobs)}", flush=True)
    print(f"[done] {done} jobs", flush=True)


if __name__ == "__main__":
    main()
