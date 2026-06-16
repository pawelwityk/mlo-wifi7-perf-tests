#!/usr/bin/env python3
"""
replot.py — regenerate 4.2.2 plots from existing CSVs without re-running sims.

Usage (from repo root):
    python3 scratch/4.2.2/replot.py
"""

import os
import sys
import importlib.util

# Locate scratch/_common for thesis_style + _robust
_HERE = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_HERE, os.pardir, "_common"))
sys.path.insert(0, _COMMON)
import thesis_style  # noqa: F401

import matplotlib
matplotlib.use("Agg")

# Import the runner module (which holds load_existing_results + plot_* functions)
_RUNNER = os.path.join(_HERE, "run_all_rerun.py")
spec = importlib.util.spec_from_file_location("_run_all_rerun", _RUNNER)
mod = importlib.util.module_from_spec(spec)
mod.__file__ = _RUNNER
spec.loader.exec_module(mod)

raw: dict = {}
done = mod.load_existing_results(raw)
print(f"Loaded {len(done)} run(s) from {mod.OUTPUT_DIR}/")

mod.plot_percentile_bands(raw)
mod.plot_results(raw)
