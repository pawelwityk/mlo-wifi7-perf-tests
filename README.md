# mlo-wifi7-perf-tests

An automated framework and test suite designed to measure and analyze the performance of Wi-Fi 7 Multi-Link Operation (MLO) under various network conditions.

This repository integrates network simulation tools with automated data processing pipelines to benchmark MLO features like throughput, latency, and link switching efficiency.

---

## Overview

Multi-Link Operation (MLO) is a flagship feature of Wi-Fi 7 (IEEE 802.11be) that allows devices to concurrently send and receive data across multiple wireless bands (2.4 GHz, 5 GHz, and 6 GHz). This project provides the tools to simulate, capture, and statistically evaluate these implementations.

### Key Features
* **Automated Simulations:** End-to-end simulation setups for Wi-Fi 7 network topologies.
* **Data Processing Pipeline:** Seamless extraction of raw simulation logs into structured CSV format.
* **Statistical Analysis:** Python-based analytical scripts to generate plots, CDFs (Cumulative Distribution Functions), and performance metrics.

---

## Data Processing Pipeline

The performance analysis follows a structured, reproducible four-step pipeline:

[ ns-3 Simulation ] ──(Simulation Results)──> [ CSV File ] ──(Input Data)──> [ Python Scripts ] ──> [ Statistical Analysis & Plots ]


1. **ns-3 Simulation:** The network scenario is executed within the network simulator, generating raw trace files.
2. **CSV Export:** Key performance indicators (KPIs) such as packet delay, jitter, and throughput are extracted into structured CSV files.
3. **Python Processing:** Local Python scripts parse the CSV data using high-performance libraries (pandas, numpy).
4. **Analysis & Visualizations:** The framework outputs production-ready charts (e.g., latency distributions, throughput over time).

---

## Prerequisites

To run the simulations and analysis scripts, ensure you have the following dependencies installed:

### Network Simulator
* **ns-3** (v3.40 or newer with Wi-Fi 7/802.11be support)
* C++17 compiler (GCC or Clang)
* CMake

### Python Environment
* Python 3.10+
* Required libraries (install via pip):
  ```bash
  pip install pandas numpy matplotlib seaborn
Usage
1. Running the Simulation
Navigate to your ns-3 directory, link the scenario from this repository, and execute:

Bash
./ns3 run "mlo-wifi7-perf-tests --simulationTime=10 --numStations=5"
This will generate the raw result files in the output directory.

2. Analyzing Results
Run the local Python scripts to process the generated CSV files and produce statistical plots:

Bash
python scripts/analyze_results.py --input data/simulation_results.csv --output charts/
Sample Visualizations
The analytical scripts automatically output several types of plots into the charts/ directory:

Throughput Stability: Time-series analysis of aggregated bandwidth across multiple links.

Latency CDF: Cumulative Distribution Function curves comparing MLO vs. Single-Link operation.

Authors
Paweł Wityk - Main Developer / Maintainer - github.com/pawelwityk

License
This project is licensed under the MIT License - see the LICENSE file for details.
